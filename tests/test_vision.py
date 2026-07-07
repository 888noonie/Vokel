import unittest

from vokel.vision import (
    CAMERA_GUIDANCE_OFFER,
    SpokenReply,
    VisualContext,
    detect_camera_command,
    should_capture_visual_context,
)


class DetectCameraCommandTest(unittest.TestCase):
    def test_shoot_triggers(self) -> None:
        for text in (
            "shoot",
            "Shoot.",
            "okay shoot",
            "take a picture",
            "take the picture please",
            "snap a photo",
            "take the shot now",
            "go on, capture this",
        ):
            self.assertEqual(detect_camera_command(text), "shoot", text)

    def test_action_and_cut(self) -> None:
        self.assertEqual(detect_camera_command("action"), "action")
        self.assertEqual(detect_camera_command("and action"), "action")
        self.assertEqual(detect_camera_command("go live"), "action")
        self.assertEqual(detect_camera_command("cut"), "cut")
        self.assertEqual(detect_camera_command("stop watching"), "cut")
        self.assertEqual(detect_camera_command("stop the live loop"), "cut")

    def test_watch_offer(self) -> None:
        self.assertEqual(detect_camera_command("watch me"), "watch")
        self.assertEqual(detect_camera_command("look at me"), "watch")
        self.assertEqual(detect_camera_command("watch this"), "watch")

    def test_strong_capture_phrase_fires_in_a_long_utterance(self) -> None:
        # The padded, polite phrasings a frustrated user actually says must still
        # fire — these leaked to the model before and were never captured.
        for text in (
            "I can see how that confused you, please take a picture",
            "okay no problem just go ahead and take a photo for me now",
            "sorry about that could you take another picture please",
        ):
            self.assertEqual(detect_camera_command(text), "shoot", text)

    def test_does_not_fire_on_ordinary_speech(self) -> None:
        for text in (
            "I cut my finger on the glass",
            "lights camera action everyone get ready",
            "what am I holding",
            "the action movie was great last night",
            "",
            "let me tell you a long story about my day at work",
        ):
            self.assertIsNone(detect_camera_command(text), text)

    def test_visual_questions_still_classified_separately(self) -> None:
        # Questions are not "commands"; they route through should_capture_visual_context.
        self.assertIsNone(detect_camera_command("what can you see"))
        self.assertTrue(should_capture_visual_context("what can you see"))


class SpokenReplyTest(unittest.TestCase):
    def test_guidance_offer_names_all_three_words(self) -> None:
        reply = SpokenReply(CAMERA_GUIDANCE_OFFER)
        lowered = reply.text.lower()
        self.assertIn("shoot", lowered)
        self.assertIn("action", lowered)
        self.assertIn("cut", lowered)

    def test_visual_context_prompt_defaults_none(self) -> None:
        self.assertIsNone(VisualContext(data_url="x", source="/dev/video0").prompt)


if __name__ == "__main__":
    unittest.main()
