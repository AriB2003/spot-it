import numpy as np
import requests
import cv2 as cv
import os
import math
import scipy.stats as sp
from design_templates import m2, urls


def template_symbols(design):
    return np.sum(design, axis=0)


def symbols_per_card(design):
    return np.max(template_symbols(design))


def total_symbols(design):
    tp = template_symbols(design)
    unique = symbols_per_card(design) * len(tp) - np.sum(tp)
    matching = np.size(design, 1)
    return unique + matching


def area_sort(face):
    return face[2] * face[3]


def fetch_images(urls):
    face_cascade_default = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # face_cascade_alternative1 = cv.CascadeClassifier(
    #     cv.data.haarcascades + "haarcascade_frontalface_alt.xml"
    # )
    face_cascade_alternative = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )
    face_cascade_profile = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_profileface.xml"
    )
    counter = 0
    for i, url in enumerate(urls):
        img_data = requests.get(url["url"]).content
        print(f"Got: {url["url"]}")
        with open("temp.png", "wb") as f:
            f.write(img_data)
        img = cv.imread("temp.png")
        os.remove("temp.png")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = []
        if url["method"] == "default":
            faces = face_cascade_default.detectMultiScale(gray, 1.3, 5)
        elif url["method"] == "alternative":
            faces = face_cascade_alternative.detectMultiScale(
                gray, url["parameters"][0], url["parameters"][1]
            )
        elif url["method"] == "profile":
            faces = face_cascade_profile.detectMultiScale(
                gray, url["parameters"][0], url["parameters"][1]
            )
        if len(faces) == 0:
            print("No faces found!")
            # exit(0)
        else:
            print(f"{len(faces)} faces found!")
        faces = faces.tolist()
        faces.sort(key=area_sort, reverse=True)
        for j, (x, y, w, h) in enumerate(faces[: url["quantity"]]):
            h = w
            offset = min(100, x, y, img.shape[1] - x - w, img.shape[0] - y - h)
            print(f"Calculated Offset: {offset}")
            roi_color = img[y - offset : y + h + offset, x - offset : x + w + offset]
            mask = np.zeros(roi_color.shape, dtype=np.uint8)
            mask2 = 255 * np.ones(roi_color.shape, dtype=np.uint8)
            # Define the center and radius of the circle
            center = (roi_color.shape[1] // 2, roi_color.shape[0] // 2)
            radius = min(roi_color.shape[1] // 2, roi_color.shape[0] // 2)
            # Draw a white circle on the mask
            cv.circle(mask, center, radius, 255, -1)
            cv.circle(mask2, center, radius, 0, -1)
            # Apply the mask to the image
            masked_image = cv.bitwise_and(roi_color, mask)
            masked_image = cv.bitwise_or(masked_image, mask2)
            counter += 1
            save_path = os.path.join(".", "source_images", f"talktuah{counter}.png")
            cv.imwrite(save_path, masked_image)
            print(f"Saved Image: {save_path}")
    return True


def resize_images(image_directory, scale):
    min_size = math.inf
    for path in image_directory:
        img = cv.imread(os.path.join(".", "source_images", path))
        min_size = min(img.shape[0], min_size)
    if scale:
        min_size *= 2
    for path in image_directory:
        fp = os.path.join(".", "source_images", path)
        img = cv.imread(fp)
        cv.imwrite(fp, cv.resize(img, (min_size, min_size)))
        print(f"Resized {path} to {(min_size, min_size)}")


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    mask = 255 * np.ones(result.shape, dtype=np.uint8)
    center = (result.shape[1] // 2, result.shape[0] // 2)
    radius = min(result.shape[1] // 2, result.shape[0] // 2)
    cv.circle(mask, center, radius, 0, -1)
    result = cv.bitwise_or(result, mask)
    return result


def generate_valid_location(distance, num_symbols):
    locations = np.array([[0, 0]], dtype=np.float32)
    angles = np.linspace(0, 2 * math.pi, num_symbols)
    for angle in angles[:-1]:
        random_distance = sp.norm.rvs(loc=distance, scale=distance / 6)
        locations = np.vstack(
            [
                locations,
                [math.cos(angle) * random_distance, math.sin(angle) * random_distance],
            ]
        )
    return locations.astype(np.float32)


def create_canvas(image_directory, num_symbols):
    shape = cv.imread(os.path.join(".", "source_images", image_directory[0])).shape
    locations = generate_valid_location(2 * shape[0], num_symbols)
    x, y, w, h = cv.boundingRect(locations)
    d = max(w, h)
    canvas = np.zeros([d, d, 3], dtype=np.uint8)
    center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    radius = min(canvas.shape[1] // 2, canvas.shape[0] // 2)
    cv.circle(canvas, center, radius, (255, 255, 255), -1)
    return canvas, locations


scale = False
# Uncomment to fetch images
# scale = fetch_images(urls)

total = total_symbols(m2)
print(f"Required Symbols: {total}")

image_directory = os.listdir(os.path.join(".", "source_images"))
print(f"Found Images:\n{"\n".join(image_directory)}")

if len(image_directory) < total:
    print("Not enough images to generate")
    exit(0)

resize_images(image_directory, scale)

canvas, locations = create_canvas(image_directory, symbols_per_card(m2))
print(locations)
cv.imwrite("hi.png", canvas)
