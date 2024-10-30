from ultralytics import YOLO
import pyautogui as auto
import cv2
import numpy as np
from yaml import safe_load
import time
import logging

logging.basicConfig(
    filename=".\\logs\\app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def backWindows():
    return *auto.size(), auto.screenshot()


def show(img) -> None:
    cv2.namedWindow("Screen")
    cv2.imshow("Screen", cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def locatePngAndMoveTo(png, pngName, locatePngAndMoveToConf) -> (int, int) or None:
    logging.info(f"Locating {pngName}")
    try:
        pngPosition = auto.locateCenterOnScreen(png, confidence=locatePngAndMoveToConf)
        logging.info(f"Move to: {pngPosition[0]},{pngPosition[1]}")
        auto.moveTo(pngPosition[0], pngPosition[1])
        auto.click(pngPosition[0], pngPosition[1])
        return pngPosition
    except auto.ImageNotFoundException:
        logging.error(f"Couldn't locate {pngName}")
        return None
    except ValueError:
        logging.warning("Screen is too small")
        return None


def locateAndDrag(arrow: np.ndarray, puzzle: np.ndarray) -> None:
    auto.click(arrow[0], arrow[1])
    auto.dragTo(puzzle[0], arrow[1], duration=1, button="left")
    logging.info(f"click {arrow[0],arrow[1]} and drag to: {puzzle[0]},{puzzle[1]}")


def main(config) -> None:
    # load the model
    model = YOLO(config["model"], task="detect")
    logging.info(f"{config['model']} Model loaded")

    screenX, screenY = auto.size()
    while True:

        # locate the blank puzzle and the arrow and
        screen = cv2.cvtColor(np.array(auto.screenshot()), cv2.COLOR_RGB2BGR)

        # detect the blank puzzle
        results = model(
            screen,
            task="detect",
            device=config["device"],
            conf=config["conf"],
            max_det=config["max_det"],
            imgsz=(screenX, screenY),
            iou=config["NMS"],
            stream=False,
        )[0].boxes

        try:
            # 获取识别结果
            puzzle_boxes = (
                results.xywh[results.cls == config["classes"]["puzzle"]]
                .cpu()
                .numpy()
                .astype(int)
            )
            arrow_box = (
                results.xywh[results.cls == config["classes"]["arrow"]]
                .cpu()
                .numpy()
                .astype(int)[0]
            )

            if len(puzzle_boxes) == 1:
                # 只识别到右边目标位置的 puzzle
                puzzle_target = puzzle_boxes[0]
                logging.info("Only detected target puzzle, proceeding")
                locateAndDrag(arrow_box, puzzle_target)

            elif len(puzzle_boxes) == 2:
                # 识别到两个 puzzle，区分滑块上方的 puzzle 和目标位置 puzzle
                puzzle_left, puzzle_right = puzzle_boxes
                # 可以通过坐标或者其他特征区分 puzzle
                if puzzle_left[0] < puzzle_right[0]:  # 假设左边的 puzzle x 坐标较小
                    puzzle_target = puzzle_right
                else:
                    puzzle_target = puzzle_left

                logging.info("Detected both puzzles, proceeding with target puzzle")
                locateAndDrag(arrow_box, puzzle_target)

            else:
                logging.info("Puzzle detection count not matched, skipping")
        except IndexError:
            logging.info("Couldn't detect anything, skipping")
        finally:
            pass

        # wait time
        logging.info(f"Waiting for {config['waitTime']}s to next detection")
        time.sleep(config["waitTime"])


if __name__ == "__main__":
    with open(".//config//runConfig.yaml", "r") as f:
        config = safe_load(f)
    for i, k in enumerate(config):
        print(k, config[k])
    main(config)
