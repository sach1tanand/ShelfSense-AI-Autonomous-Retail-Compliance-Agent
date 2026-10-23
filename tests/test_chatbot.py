"""Chat endpoint, provider, and page integration coverage."""

from __future__ import annotations

import io
import json
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

import app as web_app
from backend.chatbot import ChatProviderError, audit_context, generate_reply, load_local_env
from config.compliance_scoring import SCORING_VERSION


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()
        self.old_reports = web_app._reports.copy()
        web_app._reports.clear()

    def tearDown(self):
        web_app._reports.clear()
        web_app._reports.update(self.old_reports)

    def test_status_and_missing_key_are_explained(self):
        with patch.dict("os.environ", {"SHELFSENSE_CHAT_API_KEY": ""}):
            status = self.client.get("/api/chat")
            response = self.client.post("/api/chat", json={"message": "Explain this app"})
        self.assertFalse(status.json["configured"])
        self.assertIn("SHELFSENSE_CHAT_API_KEY", status.json["message"])
        self.assertEqual(response.status_code, 503)
        self.assertIn("not configured", response.json["error"])

    def test_invalid_request_and_missing_report_return_client_errors(self):
        self.assertEqual(self.client.post("/api/chat", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/chat", json={"message": "x" * 1501}).status_code, 400)
        missing = self.client.post("/api/chat", json={"message": "Summarize", "report_id": "missing"})
        self.assertEqual(missing.status_code, 404)

    def test_response_uses_actual_report_facts_and_excludes_image_path(self):
        web_app._reports["report-1"] = {
            "report_id": "report-1", "image_path": "private/upload/path.jpg", "image_name": "store.jpg",
            "created_at": "today", "compliance_score": {
                "version": SCORING_VERSION, "score": 82.5, "status": "partial", "coverage": 90,
            },
            "compliance": [{"shelf": "A", "row": "1", "position": "1", "sku": "A1", "status": "Needs Review"}],
            "planogram_comparison": {"summary": {"needs_review": 1}, "findings": [{
                "shelf": "A", "row": "1", "slot": "1", "expected_sku": "A1", "detected_sku": "A1",
                "status": "Needs Review", "reason": "Facing count needs review.",
            }]},
            "price_results": [],
        }
        with patch.dict("os.environ", {"SHELFSENSE_CHAT_API_KEY": "configured"}), \
             patch.object(web_app, "generate_chat_reply", return_value="One slot needs review.") as generate:
            response = self.client.post("/api/chat", json={
                "message": "What should I check?", "report_id": "report-1",
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["reply"], "One slot needs review.")
        self.assertEqual(response.json["report_id"], "report-1")
        context = generate.call_args.args[1]
        self.assertEqual(context["compliance_score"]["score"], 82.5)
        self.assertEqual(context["findings"][0]["reason"], "Facing count needs review.")
        self.assertNotIn("image_path", json.dumps(context))

    def test_context_does_not_include_raw_images_or_unnecessary_detection_data(self):
        context = audit_context("r", {"image_path": "private", "detections": [{"crop": "private"}], "compliance": []})
        self.assertNotIn("image_path", json.dumps(context))
        self.assertNotIn("detections", context)

    def test_local_env_loader_reads_key_values_without_overriding_process_env(self):
        variable = "SHELFSENSE_TEST_CHAT_ENV"
        previous = os.environ.pop(variable, None)
        try:
            with patch("pathlib.Path.read_text", return_value=f'{variable}="local-value"\n'):
                load_local_env()
            self.assertEqual(os.environ[variable], "local-value")
            with patch("pathlib.Path.read_text", return_value=f'{variable}=file-value\n'):
                load_local_env()
            self.assertEqual(os.environ[variable], "local-value")
        finally:
            os.environ.pop(variable, None)
            if previous is not None:
                os.environ[variable] = previous

    def test_provider_timeout_is_reported_cleanly(self):
        with patch.dict("os.environ", {"SHELFSENSE_CHAT_API_KEY": "secret"}), \
             patch("backend.chatbot.urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(ChatProviderError, "timed out"):
                generate_reply("Help", {"available": False})

    def test_provider_failure_is_returned_as_gateway_error(self):
        with patch.dict("os.environ", {"SHELFSENSE_CHAT_API_KEY": "configured"}), \
             patch.object(web_app, "generate_chat_reply", side_effect=ChatProviderError("Provider unavailable.")):
            response = self.client.post("/api/chat", json={"message": "Help"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json["error"], "Provider unavailable.")

    def test_provider_http_error_includes_safe_diagnostic_message(self):
        error_body = json.dumps({"error": {"message": "Model not found for generateContent."}}).encode()
        http_error = HTTPError("https://provider.invalid", 404, "Not Found", {}, io.BytesIO(error_body))
        with patch.dict("os.environ", {"SHELFSENSE_CHAT_API_KEY": "secret-test-key"}), \
             patch("backend.chatbot.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(ChatProviderError, "Model not found for generateContent"):
                generate_reply("Help", {"available": False})

    def test_provider_success_and_widgets_on_shared_and_dashboard_pages(self):
        response_body = {"choices": [{"message": {"content": "Audit summary."}}]}
        with patch.dict("os.environ", {
            "SHELFSENSE_CHAT_API_KEY": "secret", "SHELFSENSE_CHAT_MODEL": "test-model",
            "SHELFSENSE_CHAT_API_URL": "https://api.openai.com/v1/chat/completions",
        }), patch("backend.chatbot.urlopen", return_value=io.BytesIO(json.dumps(response_body).encode())) as call:
            answer = generate_reply("Summarize", {"available": False})
        self.assertEqual(answer, "Audit summary.")
        self.assertEqual(call.call_args.kwargs["timeout"], 20)
        request_body = json.loads(call.call_args.args[0].data)
        system_prompt = request_body["messages"][0]["content"]
        self.assertIn("short headings, bullet lists, and bold labels", system_prompt)
        self.assertIn("Confirmed findings, Needs Review, and Recommended", system_prompt)

        upload = self.client.get("/upload")
        dashboard = self.client.get("/")
        for response in (upload, dashboard):
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"shelf-chat-launcher", response.data)
        css = self.client.get("/static/chatbot.css")
        self.assertIn(b"position:fixed", css.data.replace(b" ", b""))
        self.assertIn(b"max-width:520px", css.data.replace(b" ", b""))
        self.assertIn(b"bottom:20px", css.data.replace(b" ", b""))
        css.close()
        script = self.client.get("/static/chatbot.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn(b"requestSubmit", script.data)
        self.assertIn(b"renderAssistantMarkdown", script.data)
        self.assertIn(b"escapeHtml", script.data)
        script.close()

    def test_gemini_endpoint_uses_google_auth_and_generate_content_schema(self):
        response_body = {"candidates": [{"content": {"parts": [{"text": "Gemini answer."}]}}]}
        with patch.dict("os.environ", {
            "SHELFSENSE_CHAT_API_KEY": "google-test-key",
            "SHELFSENSE_CHAT_API_URL": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        }), patch("backend.chatbot.urlopen", return_value=io.BytesIO(json.dumps(response_body).encode())) as call:
            answer = generate_reply("Summarize", {"available": False})

        request = call.call_args.args[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        body = json.loads(request.data)
        self.assertEqual(answer, "Gemini answer.")
        self.assertIn("gemini-3.5-flash-lite:generateContent", request.full_url)
        self.assertEqual(headers["x-goog-api-key"], "google-test-key")
        self.assertNotIn("authorization", headers)
        self.assertIn("contents", body)
        self.assertIn("systemInstruction", body)


if __name__ == "__main__":
    unittest.main()
