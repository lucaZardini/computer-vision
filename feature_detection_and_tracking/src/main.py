from manager import TrackManager


def main():
    TrackManager.track_video("../video/video.mp4", online=True, save_video=False)


if __name__ == "__main__":
    main()
