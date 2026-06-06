import argparse

from vehicle_entry import start_live_vehicle_entry


DEFAULT_VIDEO_SOURCE = "https://ik.imagekit.io/6f8hdxg1w/Screen%20Recording%202026-06-03%20092635.mp4"


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Smoke test for vehicle entry recognition")
	parser.add_argument("--source", default=DEFAULT_VIDEO_SOURCE, help="Camera index, file path, or video URL")
	parser.add_argument("--skip", type=int, default=5, help="Process every Nth frame")
	parser.add_argument("--max", type=int, default=10, help="Stop after this many processed frames")
	parser.add_argument("--show-window", action="store_true", help="Display the OpenCV window")

	args = parser.parse_args()

	source = args.source
	try:
		source = int(source)
	except Exception:
		pass

	start_live_vehicle_entry(
		source=source,
		frame_skip=args.skip,
		max_frames=args.max,
		show_window=args.show_window,
	)
