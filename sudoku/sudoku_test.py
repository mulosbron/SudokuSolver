import os
import random
import cv2
import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = '../model/sudoku/model.keras'
model = load_model(MODEL_PATH)

SUDOKU_IMG_DIR = 'sudoku_img'

SOLVED_SUDOKU_DIR = 'solved_sudoku'
PROCESSING_STEPS_DIR = 'processing_steps'
os.makedirs(SOLVED_SUDOKU_DIR, exist_ok=True)
os.makedirs(PROCESSING_STEPS_DIR, exist_ok=True)

IMG_SIZE = 100


def get_contours(img, original_img, save_dir, image_name):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 60000:
            print(f"Found contour area: {area}")
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area
    if biggest is not None:
        cv2.drawContours(original_img, [biggest], -1, (0, 255, 0), 2)
        contour_image_path = os.path.join(save_dir, f"{image_name}_contours.png")
        cv2.imwrite(contour_image_path, original_img)
        print(f"Contours drawn and saved: {contour_image_path}")

        pts = biggest.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        width, height = 900, 900

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        img_perspective = cv2.warpPerspective(original_img, M, (width, height))
        perspective_image_path = os.path.join(save_dir, f"{image_name}_perspective.png")
        cv2.imwrite(perspective_image_path, img_perspective)
        print(f"Perspective transformed image saved: {perspective_image_path}")

        img_corners = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)
        _, img_corners_bin = cv2.threshold(img_corners, 128, 255, cv2.THRESH_BINARY_INV)
        binary_image_path = os.path.join(save_dir, f"{image_name}_binary.png")
        cv2.imwrite(binary_image_path, img_corners_bin)
        print(f"Binary image saved: {binary_image_path}")

        return img_corners_bin, img_perspective
    return None, None


def classify(img, model, save_dir, image_name):
    crop_val = 20
    digits_list = []
    classified_digits_image = np.ones((900, 900, 3), dtype=np.uint8) * 255

    cell_size = img.shape[0] // 9
    for i in range(9):
        for j in range(9):
            y_start = i * cell_size + crop_val
            y_end = (i + 1) * cell_size - crop_val
            x_start = j * cell_size + crop_val
            x_end = (j + 1) * cell_size - crop_val
            cell = img[y_start:y_end, x_start:x_end]
            cell_resized = cv2.resize(cell, (IMG_SIZE, IMG_SIZE))
            cell_array = cell_resized.astype('float32') / 255.0
            cell_array = cell_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

            prediction = model.predict(cell_array, verbose=0)
            predicted_class = np.argmax(prediction)
            prob = np.max(prediction)

            if prob > 0.8:
                digit = predicted_class
                color = (0, 0, 0)
            else:
                digit = 0
                color = (255, 255, 255)

            digits_list.append(digit)

            if digit != 0:
                cv2.putText(classified_digits_image, str(digit),
                            (j * cell_size + 25, i * cell_size + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)

    classified_image_path = os.path.join(save_dir, f"{image_name}_classified.png")
    cv2.imwrite(classified_image_path, classified_digits_image)
    print(f"Classified digits saved: {classified_image_path}")

    return digits_list


def solve(grid):
    find = find_empty(grid)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if is_valid(grid, i, (row, col)):
            grid[row][col] = i

            if solve(grid):
                return True

            grid[row][col] = 0
    return False


def is_valid(grid, num, coordinate):
    for i in range(len(grid[0])):
        if grid[coordinate[0]][i] == num and coordinate[1] != i:
            return False

    for i in range(len(grid)):
        if grid[i][coordinate[1]] == num and coordinate[0] != i:
            return False

    box_x = coordinate[1] // 3
    box_y = coordinate[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if grid[i][j] == num and (i, j) != coordinate:
                return False
    return True


def find_empty(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                return (i, j)
    return None


def save_sudoku(sudoku2d, sudoku2d_unsolved, solved_cell_path):
    solved_cell = np.ones((900, 900, 3), dtype=np.uint8) * 255
    for i in range(8):
        cv2.line(solved_cell, ((i + 1) * 100, 0), ((i + 1) * 100, 900), (0, 0, 0), 2)
        cv2.line(solved_cell, (0, (i + 1) * 100), (900, (i + 1) * 100), (0, 0, 0), 2)
    for i in range(2):
        cv2.line(solved_cell, ((i + 1) * 300, 0), ((i + 1) * 300, 900), (0, 0, 0), 4)
        cv2.line(solved_cell, (0, (i + 1) * 300), (900, (i + 1) * 300), (0, 0, 0), 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    thickness = 4
    for (index_row, row) in enumerate(sudoku2d):
        for (index_num, num) in enumerate(row):
            pos = (index_num * 100 + 25, index_row * 100 + 70)
            color = (0, 0, 0)
            if sudoku2d_unsolved[index_row][index_num] == 0:
                color = (0, 0, 255)

            cv2.putText(solved_cell, str(num), pos, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(solved_cell_path, solved_cell)
    print(f"Solved Sudoku saved: {solved_cell_path}")


def process_sudoku_image(img_path, model):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image could not be read: {img_path}")
        return

    image_processing_dir = os.path.join(PROCESSING_STEPS_DIR, img_name)
    os.makedirs(image_processing_dir, exist_ok=True)

    original_image_path = os.path.join(image_processing_dir, f"{img_name}_original.png")
    cv2.imwrite(original_image_path, img)
    print(f"Original image saved: {original_image_path}")

    original_img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_image_path = os.path.join(image_processing_dir, f"{img_name}_grayscale.png")
    cv2.imwrite(grayscale_image_path, img_gray)
    print(f"Grayscale image saved: {grayscale_image_path}")

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 3)
    blurred_image_path = os.path.join(image_processing_dir, f"{img_name}_blurred.png")
    cv2.imwrite(blurred_image_path, img_blur)
    print(f"Blurred image saved: {blurred_image_path}")

    img_canny = cv2.Canny(img_blur, 50, 50)
    canny_image_path = os.path.join(image_processing_dir, f"{img_name}_canny.png")
    cv2.imwrite(canny_image_path, img_canny)
    print(f"Canny Edge Detection applied and saved: {canny_image_path}")

    img_contours_bin, img_perspective = get_contours(img_canny, original_img, image_processing_dir, img_name)
    if img_contours_bin is not None:
        sudoku_digits = classify(img_contours_bin, model, image_processing_dir, img_name)
        if sudoku_digits:
            sudoku2d = [sudoku_digits[i * 9:(i + 1) * 9] for i in range(9)]
            sudoku2d_unsolved = [row.copy() for row in sudoku2d]

            if solve(sudoku2d):
                print(f"Sudoku solved: {os.path.basename(img_path)}")
                solved_cell_path = os.path.join(SOLVED_SUDOKU_DIR, f"solved_{os.path.basename(img_path)}")
                save_sudoku(sudoku2d, sudoku2d_unsolved, solved_cell_path)
            else:
                print(f"Sudoku could not be solved: {os.path.basename(img_path)}")
        else:
            print(f"No digits detected in the image: {os.path.basename(img_path)}")
    else:
        print(f"Sudoku grid could not be detected in the image: {os.path.basename(img_path)}")


def test_sudoku():
    image_files = [f for f in os.listdir(SUDOKU_IMG_DIR) if os.path.isfile(os.path.join(SUDOKU_IMG_DIR, f))]
    if not image_files:
        print(f"No Sudoku images found in the {SUDOKU_IMG_DIR} directory.")
        return

    num_images_to_process = max(1, len(image_files))
    selected_images = random.sample(image_files, num_images_to_process)

    print(f"Images to be processed ({len(selected_images)}): {selected_images}")

    for img_name in selected_images:
        img_path = os.path.join(SUDOKU_IMG_DIR, img_name)
        process_sudoku_image(img_path, model)


if __name__ == "__main__":
    test_sudoku()
