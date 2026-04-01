import cv2
import os
import json
import argparse
from tqdm import trange


def create_video(images_folder, output_video, fps, font_scale, text_color, text_position):
    images = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front')) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    frame = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i in trange(1, len(images)):
        image = images[i]
        f = open(os.path.join(images_folder, f'meta/{i:04}.json'), 'r')
        meta = json.load(f)
        steer = float(meta['steer'])
        throttle = float(meta['throttle'])
        brake = float(meta['brake'])
        # command = float(meta['command'])
        # command_list = ["VOID", "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT",  "CHANGE LANE RIGHT",]
        speed = float(meta['speed'])
        text = f'speed: {round(speed,2)}, steer: {round(steer,2)}, throttle: {round(throttle,2)}, brake: {round(brake,2)}'#, command: {command_list[int(command)]}'
        img = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), image))
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
        video.write(img)
    video.release()


def is_route_dir(path):
    return os.path.isdir(os.path.join(path, "rgb_front")) and os.path.isdir(os.path.join(path, "meta"))


def batch_create_videos(input_dir, output_dir, fps, font_scale, text_color, text_position):
    route_dirs = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isdir(path) and is_route_dir(path):
            route_dirs.append(path)

    if not route_dirs:
        raise RuntimeError(f"No valid route folders found under: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    for route_dir in route_dirs:
        route_name = os.path.basename(route_dir)
        output_video = os.path.join(output_dir, f"{route_name}.mp4")
        print(f"[generate_video] processing {route_name} -> {output_video}")
        create_video(
            images_folder=route_dir,
            output_video=output_video,
            fps=fps,
            font_scale=font_scale,
            text_color=text_color,
            text_position=text_position,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one video from a route folder, or batch-generate videos from a parent directory."
    )
    parser.add_argument(
        "-f",
        "--images-folder",
        help="Route folder containing rgb_front/ and meta/ subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--output-video",
        help="Output mp4 file path for single-route mode.",
    )
    parser.add_argument(
        "--input-dir",
        help="Parent directory containing multiple route folders for batch mode.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for batch mode.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Video FPS.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Overlay font scale.")
    parser.add_argument("--text-x", type=int, default=50, help="Overlay text x position.")
    parser.add_argument("--text-y", type=int, default=50, help="Overlay text y position.")
    args = parser.parse_args()

    single_mode = args.images_folder or args.output_video
    batch_mode = args.input_dir or args.output_dir

    if single_mode and batch_mode:
        parser.error("Use either single mode (-f/-o) or batch mode (--input-dir/--output-dir), not both.")
    if not single_mode and not batch_mode:
        parser.error("You must provide either single mode (-f/-o) or batch mode (--input-dir/--output-dir).")
    if single_mode and (not args.images_folder or not args.output_video):
        parser.error("Single mode requires both --images-folder and --output-video.")
    if batch_mode and (not args.input_dir or not args.output_dir):
        parser.error("Batch mode requires both --input-dir and --output-dir.")

    return args


if __name__ == "__main__":
    args = parse_args()
    text_color = (255, 255, 255)
    text_position = (args.text_x, args.text_y)

    if args.input_dir:
        batch_create_videos(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            font_scale=args.font_scale,
            text_color=text_color,
            text_position=text_position,
        )
    else:
        create_video(
            images_folder=args.images_folder,
            output_video=args.output_video,
            fps=args.fps,
            font_scale=args.font_scale,
            text_color=text_color,
            text_position=text_position,
        )
