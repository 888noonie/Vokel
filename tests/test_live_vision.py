import unittest
from pathlib import Path

from scripts.live_vision import build_parser
from vokel.vision import (
    CameraDevice,
    build_capture_command,
    build_chat_payload,
    capture_prompt_for,
    is_loopback_url,
    pick_default_camera,
    should_capture_visual_context,
)


class LiveVisionTests(unittest.TestCase):
    def test_parser_defaults_to_ps3_eye_path(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.device, "/dev/video4")
        self.assertEqual(args.count, 0)

    def test_capture_command_uses_requested_device_and_shape(self) -> None:
        command = build_capture_command(
            device="/dev/video7",
            output_pattern=Path("/tmp/frame-%02d.jpg"),
            width=320,
            height=240,
            framerate=60,
            warmup_frames=3,
        )

        self.assertIn("device=/dev/video7", command)
        self.assertIn("num-buffers=3", command)
        # Format-agnostic: decodebin negotiates the camera's native format (YUY2, MJPG,
        # or the GREY an IR cam emits) and videoscale fits the requested shape.
        self.assertIn("decodebin", command)
        self.assertIn("video/x-raw,width=320,height=240", command)

    def test_payload_places_image_before_text_for_gemma(self) -> None:
        payload = build_chat_payload(
            model="local-vlm",
            image_bytes=b"jpeg",
            prompt="Describe this.",
            max_tokens=42,
        )

        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[0]["image_url"]["url"], "data:image/jpeg;base64,anBlZw==")
        self.assertEqual(content[1], {"type": "text", "text": "Describe this."})
        self.assertEqual(payload["max_tokens"], 42)

    def test_pick_default_camera_prefers_integrated_webcam(self) -> None:
        cameras = [
            CameraDevice(path="/dev/video0", name="Integrated Camera"),
            CameraDevice(path="/dev/video4", name="gspca main driver"),
        ]
        self.assertEqual(pick_default_camera(cameras), "/dev/video0")

    def test_pick_default_camera_avoids_ir_node_when_color_exists(self) -> None:
        cameras = [
            CameraDevice(path="/dev/video2", name="Built-in webcam (IR)"),
            CameraDevice(path="/dev/video4", name="gspca main driver"),
        ]
        # The IR node only emits grayscale infrared, useless for visual questions.
        self.assertEqual(pick_default_camera(cameras), "/dev/video4")

    def test_pick_default_camera_falls_back_to_ir_when_only_option(self) -> None:
        cameras = [CameraDevice(path="/dev/video2", name="Built-in webcam (IR)")]
        self.assertEqual(pick_default_camera(cameras), "/dev/video2")

    def test_capture_prompt_uses_clean_prompt_for_bare_trigger(self) -> None:
        for text in ("shoot", "take a picture", "snap a photo please"):
            self.assertEqual(capture_prompt_for(text, bare_prompt="GENERIC"), "GENERIC", text)

    def test_capture_prompt_keeps_a_real_question(self) -> None:
        text = "take a picture and tell me what's on top of my head"
        self.assertEqual(capture_prompt_for(text, bare_prompt="GENERIC"), text)

    def test_loopback_check_protects_camera_frames_by_default(self) -> None:
        self.assertTrue(is_loopback_url("http://127.0.0.1:1234/v1/chat/completions"))
        self.assertTrue(is_loopback_url("http://localhost:1234/v1/chat/completions"))
        self.assertFalse(is_loopback_url("https://example.com/v1/chat/completions"))

    def test_visual_context_detection_matches_grounded_camera_questions(self) -> None:
        self.assertTrue(should_capture_visual_context("What am I holding?"))
        self.assertTrue(should_capture_visual_context("Can you see what is in my hand?"))
        self.assertTrue(should_capture_visual_context("Take a look at this."))
        self.assertFalse(should_capture_visual_context("Show me an image of a camera."))
        self.assertFalse(should_capture_visual_context("Tell me something interesting."))


if __name__ == "__main__":
    unittest.main()
