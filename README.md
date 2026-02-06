import os
import io
import traceback
from tkinter import Tk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError
from rembg import remove

ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def refine_edges(image):
    """Kanten verfeinern für sauberere Freisteller."""
    try:
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
        mask = cv2.inRange(cv_img, (0, 0, 0, 1), (255, 255, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (1, 1), 0)
        result = cv2.bitwise_and(cv_img, cv_img, mask=mask)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    except Exception as exc:
        print(f"Fehler beim Verfeinern der Kanten: {exc}")
        traceback.print_exc()
        return image


def adaptive_shadow_correction(image, clip_limit=1.2, blend_ratio=0.7):
    """Adaptive Schattenkorrektur."""
    try:
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
        corrected = cv2.merge((cl, a, b))
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)
        return Image.blend(image, Image.fromarray(corrected), blend_ratio)
    except Exception as exc:
        print(f"Fehler bei der adaptiven Schattenkorrektur: {exc}")
        traceback.print_exc()
        return image


def reduce_reflections(image, highlight_threshold=230, blur_strength=5):
    """Reflexionen reduzieren."""
    try:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, highlight_threshold, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(bgr, (blur_strength, blur_strength), 0)
        bgr[mask == 255] = blurred[mask == 255]
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    except Exception as exc:
        print(f"Fehler bei der Reflexionsminderung: {exc}")
        traceback.print_exc()
        return image


def auto_adjust_image(image):
    """Automatische Anpassungen: Reflexionen, Schatten, Helligkeit, Kontrast, Farben, Schärfe."""
    try:
        image = reduce_reflections(image, highlight_threshold=230, blur_strength=5)
        image = adaptive_shadow_correction(image, clip_limit=1.2, blend_ratio=0.7)
        image = ImageEnhance.Brightness(image).enhance(1.2)
        image = ImageEnhance.Contrast(image).enhance(1.1)
        image = ImageEnhance.Color(image).enhance(1.1)
        return ImageEnhance.Sharpness(image).enhance(1.5)
    except Exception as exc:
        print(f"Fehler bei der automatischen Anpassung: {exc}")
        traceback.print_exc()
        return image


def process_image(input_path, output_path):
    """Einzelbild verarbeiten: Hintergrund entfernen + Anpassungen."""
    try:
        with open(input_path, "rb") as file:
            output_bytes = remove(file.read())
        image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        if not image.split()[-1].getbbox():
            print(f"Kein Objekt erkannt in {input_path}. Überspringe Datei.")
            return
        refined = refine_edges(image)
        white_bg = Image.new("RGBA", refined.size, (255, 255, 255, 255))
        solid = Image.alpha_composite(white_bg, refined).convert("RGB")
        auto_adjust_image(solid).save(output_path, format="JPEG", quality=95)
    except UnidentifiedImageError:
        print(f"Ungültige Bilddatei: {input_path}")
    except Exception as exc:
        print(f"Fehler bei der Verarbeitung von {input_path}: {exc}")
        traceback.print_exc()


def set_white_background_folder(input_folder, output_folder):
    """Alle Bilder eines Ordners verarbeiten."""
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(ALLOWED_EXTS):
            continue
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"white_bg_{base_name}.jpg")
        process_image(input_path, output_path)


def choose_folder(title):
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected


def main():
    try:
        messagebox.showinfo(
            "Willkommen",
            "Dieses Tool entfernt Hintergründe und passt Reflexionen/Schatten automatisch an.",
        )
        input_folder = choose_folder("Wähle den Eingabeordner aus")
        if not input_folder:
            messagebox.showinfo("Info", "Kein Eingabeordner ausgewählt. Beende Programm.")
            return
        output_folder = choose_folder("Wähle den Ausgabeordner aus")
        if not output_folder:
            messagebox.showinfo("Info", "Kein Ausgabeordner ausgewählt. Beende Programm.")
            return
        set_white_background_folder(input_folder, output_folder)
        messagebox.showinfo("Fertig", f"Alle Bilder wurden verarbeitet und gespeichert in:\n{output_folder}")
    except Exception as exc:
        messagebox.showerror("Fehler", f"Ein schwerwiegender Fehler ist aufgetreten:\n{exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
