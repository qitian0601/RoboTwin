from pathlib import Path

from script.camera_eval_resume import retain_episode_video


def test_failure_video_is_retained(tmp_path: Path) -> None:
    video = tmp_path / "failure.mp4"
    video.write_bytes(b"video")

    retained = retain_episode_video(video, success=False, failures_only=True)

    assert retained == video
    assert video.read_bytes() == b"video"


def test_success_video_is_deleted_in_failures_only_mode(tmp_path: Path) -> None:
    video = tmp_path / "success.mp4"
    video.write_bytes(b"video")

    retained = retain_episode_video(video, success=True, failures_only=True)

    assert retained is None
    assert not video.exists()


def test_normal_recording_keeps_success_video(tmp_path: Path) -> None:
    video = tmp_path / "success.mp4"
    video.write_bytes(b"video")

    retained = retain_episode_video(video, success=True, failures_only=False)

    assert retained == video
    assert video.exists()
