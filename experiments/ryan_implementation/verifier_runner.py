# verifier_runner.py
import glob
import sys
from pathlib import Path

import cv2

from vertical_corridor import VerticalCorridorVerifier
from controlled_descent import ControlledDescentVerifier
from pitch_trimming import PitchTrimmingVerifier
from final_success import FinalSuccessVerifier


def run_on_image(path: Path):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"{path}: could not load image")
        return

    v_corr = VerticalCorridorVerifier()
    v_desc = ControlledDescentVerifier()
    v_pitch = PitchTrimmingVerifier()
    v_final = FinalSuccessVerifier()

    vc_ok = v_corr(frame)
    cd_ok = v_desc(frame)
    pt_ok = v_pitch(frame)
    fs_ok = v_final(frame)

    print(f"\nImage: {path}")
    print(f"  VerticalCorridor:   {vc_ok}")
    print(f"  ControlledDescent:  {cd_ok}")
    print(f"  PitchTrimming:      {pt_ok}")
    print(f"  FinalSuccess:       {fs_ok}")


def main():
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(Path("screenshots").glob("*.png"))

    if not paths:
        print("No images provided and no screenshots/*.png found.")
        return

    for p in paths:
        run_on_image(p)


if __name__ == "__main__":
    main()
