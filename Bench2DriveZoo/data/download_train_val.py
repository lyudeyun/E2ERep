#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path


REPO_ID = "rethinklab/Bench2Drive"
REPO_TYPE = "dataset"

VAL_CLIPS = [
    "ParkingCrossingPedestrian_Town13_Route545_Weather25",
    "OppositeVehicleTakingPriority_Town04_Route214_Weather6",
    "DynamicObjectCrossing_Town02_Route11_Weather11",
    "AccidentTwoWays_Town12_Route1115_Weather23",
    "VehicleTurningRoute_Town15_Route504_Weather10",
    "ParkingExit_Town12_Route922_Weather12",
    "SignalizedJunctionLeftTurn_Town04_Route173_Weather26",
    "EnterActorFlow_Town03_Route132_Weather2",
    "HighwayExit_Town06_Route312_Weather0",
    "VanillaSignalizedTurnEncounterRedLight_Town15_Route491_Weather23",
    "CrossingBicycleFlow_Town12_Route977_Weather15",
    "OppositeVehicleRunningRedLight_Town04_Route180_Weather23",
    "VanillaSignalizedTurnEncounterRedLight_Town07_Route359_Weather21",
    "ParkingCutIn_Town13_Route1343_Weather1",
    "ParkedObstacle_Town06_Route282_Weather22",
    "TJunction_Town06_Route306_Weather20",
    "PedestrianCrossing_Town13_Route747_Weather19",
    "VehicleTurningRoutePedestrian_Town15_Route445_Weather11",
    "ConstructionObstacle_Town12_Route78_Weather0",
    "HazardAtSideLaneTwoWays_Town12_Route1151_Weather7",
    "ControlLoss_Town04_Route170_Weather14",
    "MergerIntoSlowTrafficV2_Town12_Route857_Weather25",
    "DynamicObjectCrossing_Town01_Route3_Weather3",
    "SignalizedJunctionRightTurn_Town03_Route118_Weather14",
    "BlockedIntersection_Town03_Route135_Weather5",
    "MergerIntoSlowTraffic_Town06_Route317_Weather5",
    "NonSignalizedJunctionRightTurn_Town03_Route126_Weather18",
    "ParkedObstacleTwoWays_Town13_Route1333_Weather26",
    "ConstructionObstacleTwoWays_Town12_Route1093_Weather1",
    "TJunction_Town05_Route260_Weather0",
    "NonSignalizedJunctionLeftTurn_Town07_Route342_Weather3",
    "HighwayCutIn_Town12_Route1029_Weather15",
    "HazardAtSideLane_Town10HD_Route373_Weather9",
    "YieldToEmergencyVehicle_Town04_Route166_Weather10",
    "HardBreakRoute_Town01_Route32_Weather6",
    "SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2",
    "ConstructionObstacle_Town10HD_Route74_Weather22",
    "ControlLoss_Town10HD_Route378_Weather14",
    "Accident_Town05_Route218_Weather10",
    "InterurbanActorFlow_Town12_Route1291_Weather1",
    "LaneChange_Town06_Route307_Weather21",
    "InvadingTurn_Town02_Route95_Weather9",
    "VanillaNonSignalizedTurnEncounterStopsign_Town12_Route979_Weather9",
    "StaticCutIn_Town05_Route226_Weather18",
    "VehicleOpensDoorTwoWays_Town12_Route1203_Weather7",
    "VehicleTurningRoutePedestrian_Town15_Route481_Weather19",
    "VanillaSignalizedTurnEncounterGreenLight_Town07_Route354_Weather8",
    "NonSignalizedJunctionLeftTurnEnterFlow_Town12_Route949_Weather13",
    "InterurbanAdvancedActorFlow_Town06_Route324_Weather2",
    "ParkedObstacle_Town10HD_Route372_Weather8",
]


def _require_hf():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # noqa: F401
    except Exception:
        print("huggingface_hub not available. Please install it in your conda env:")
        print("  python -m pip install huggingface_hub")
        sys.exit(1)


def _hf_hub_download_to_dir(filename: str, out_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except TypeError:
        cached = hf_hub_download(
            REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            resume_download=True,
        )
        dest = out_dir / filename
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, dest)
        return str(dest)


def download_all(out_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Bench2Drive clips."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="val",
        choices=["val", "all"],
        help="val: 50 validation clips; all: full dataset",
    )
    args = parser.parse_args()

    out_dir = Path("Bench2Drive-all" if args.mode == "all" else "Bench2Drive-val")
    out_dir.mkdir(parents=True, exist_ok=True)

    _require_hf()

    if args.mode == "all":
        print(f"Downloading full dataset to: {out_dir} ...")
        download_all(out_dir)
        print(f"Done: {out_dir}")
        return 0

    for clip in VAL_CLIPS:
        fname = f"{clip}.tar.gz"
        print(f"Downloading {fname} ...")
        _hf_hub_download_to_dir(fname, out_dir)
        print(f"Done: {fname}")

    print(f"All validation clips downloaded to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
