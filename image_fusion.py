import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageChops, ImageEnhance
import numpy as np
import cv2
import pywt

class ImageMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Програма комплексування зображень")
        
        # Ініціалізація змінних зображень
        self.image1 = None 
        self.image2 = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Створюємо верхній фрейм для зображень та результату
        top_frame = tk.Frame(self.root)
        top_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Полотна для відображення зображень з підписами
        frame1 = tk.Frame(top_frame)
        frame1.grid(row=0, column=0, padx=5, pady=5)
        tk.Label(frame1, text="Натисніть, щоб завантажити перше зображення").pack()
        self.canvas1 = tk.Canvas(frame1, width=640, height=480, bg="gray", cursor="hand2")
        self.canvas1.pack()

        # Полотно для відображення результату
        result_frame = tk.Frame(top_frame)
        result_frame.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(result_frame, text="Натисніть, щоб зберегти результат").pack()
        self.result_canvas = tk.Canvas(result_frame, width=640, height=480, bg="gray", cursor="hand2")
        self.result_canvas.pack()

        # Фрейм для другого зображення
        frame2 = tk.Frame(top_frame)
        frame2.grid(row=1, column=0, padx=5, pady=5)
        tk.Label(frame2, text="Натисніть, щоб завантажити друге зображення").pack()
        self.canvas2 = tk.Canvas(frame2, width=640, height=480, bg="gray", cursor="hand2")
        self.canvas2.pack()

        # Додаємо обробники кліків
        self.canvas1.bind("<Button-1>", lambda e: self.load_image1())
        self.canvas2.bind("<Button-1>", lambda e: self.load_image2())
        self.result_canvas.bind("<Button-1>", lambda e: self.save_result())

        # Створюємо фрейм для елементів керування
        controls_frame = tk.Frame(top_frame)
        controls_frame.grid(row=1, column=1, padx=5, pady=5, sticky="n")

        # Випадаючий список для вибору методу злиття
        tk.Label(controls_frame, text="Метод злиття:").grid(row=0, column=0, padx=5, pady=5)
        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(controls_frame, textvariable=self.method_var)
        self.method_combobox["values"] = (
            "Додавання", 
            "Множення", 
            "Середнє", 
            "Різниця",
            "Максимум",
            "Мінімум",
            "Зважене додавання",
            "Степеневе перетворення",
            "Логарифмічне перетворення",
            "Метод головних компонент (PCA)",
            "Вейвлет перетворення",
            "Піраміда Лапласа",
            "Маска прозорості",
            "Градієнтне злиття",
            "Метод локальної варіації"
        )
        self.method_combobox.grid(row=1, column=0, padx=5, pady=5)
        self.method_combobox.current(0)
        self.method_combobox.bind('<<ComboboxSelected>>', self.on_method_change)

        # Повзунок для налаштування ваги (для зваженого додавання)
        self.weight_label = tk.Label(controls_frame, text="Вага першого зображення:")
        self.weight_label.grid(row=2, column=0, padx=5, pady=5)
        self.weight_scale = tk.Scale(controls_frame, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL)
        self.weight_scale.set(0.5)
        self.weight_scale.grid(row=3, column=0, padx=5, pady=5)
        
        # Початково приховуємо повзунок
        self.weight_label.grid_remove()
        self.weight_scale.grid_remove()

        # Кнопка для злиття зображень
        tk.Button(controls_frame, text="Об'єднати зображення", command=self.merge_images).grid(row=4, column=0, padx=5, pady=20)

    def on_method_change(self, event):
        method = self.method_var.get()
        if method == "Зважене додавання":
            self.weight_label.grid()
            self.weight_scale.grid()
        else:
            self.weight_label.grid_remove()
            self.weight_scale.grid_remove()

    def load_image1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Файли зображень", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image1 = Image.open(file_path).resize((640, 480)).convert('RGB')
            self.display_image(self.image1, self.canvas1)

    def load_image2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Файли зображень", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image2 = Image.open(file_path).resize((640, 480)).convert('RGB')
            self.display_image(self.image2, self.canvas2)

    def display_image(self, image, canvas):
        tk_image = ImageTk.PhotoImage(image)
        canvas.image = tk_image
        # Очищаємо полотно перед відображенням нового зображення
        canvas.delete("all")
        # Відображаємо зображення з прив'язкою до північно-західного кута (0,0) та розміром полотна
        canvas.create_image(0, 0, anchor='nw', image=tk_image)

    def to_numpy(self, image):
        return np.array(image)

    def to_pil(self, array):
        return Image.fromarray(np.uint8(array))

    def save_result(self):
        if not hasattr(self, 'merged_image'):
            messagebox.showerror("Помилка", "Спочатку виконайте комплексування зображень")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG файли", "*.png"),
                ("JPEG файли", "*.jpg"),
                ("Всі файли", "*.*")
            ]
        )
        if file_path:
            try:
                self.merged_image.save(file_path)
                messagebox.showinfo("Успіх", "Результат успішно збережено")
            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка при збереженні: {str(e)}")

    def merge_images(self):
        if self.image1 and self.image2:
            method = self.method_var.get()
            weight = self.weight_scale.get()

            try:
                if method == "Додавання":
                    self.merged_image = ImageChops.add(self.image1, self.image2, scale=2.0)
                
                elif method == "Множення":
                    self.merged_image = ImageChops.multiply(self.image1, self.image2)
                
                elif method == "Середнє":
                    self.merged_image = Image.blend(self.image1, self.image2, alpha=0.5)
                
                elif method == "Різниця":
                    self.merged_image = ImageChops.difference(self.image1, self.image2)
                
                elif method == "Максимум":
                    img1_array = self.to_numpy(self.image1)
                    img2_array = self.to_numpy(self.image2)
                    merged_array = np.maximum(img1_array, img2_array)
                    self.merged_image = self.to_pil(merged_array)
                
                elif method == "Мінімум":
                    img1_array = self.to_numpy(self.image1)
                    img2_array = self.to_numpy(self.image2)
                    merged_array = np.minimum(img1_array, img2_array)
                    self.merged_image = self.to_pil(merged_array)
                
                elif method == "Зважене додавання":
                    self.merged_image = Image.blend(self.image1, self.image2, weight)
                
                elif method == "Степеневе перетворення":
                    img1_array = self.to_numpy(self.image1)
                    img2_array = self.to_numpy(self.image2)
                    merged_array = np.power(img1_array, 0.6) * np.power(img2_array, 0.4)
                    self.merged_image = self.to_pil(merged_array)
                
                elif method == "Логарифмічне перетворення":
                    img1_array = self.to_numpy(self.image1)
                    img2_array = self.to_numpy(self.image2)
                    merged_array = np.log1p(img1_array + img2_array)
                    merged_array = (merged_array / merged_array.max() * 255)
                    self.merged_image = self.to_pil(merged_array)
                
                elif method == "Метод головних компонент (PCA)":
                    img1_array = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2GRAY)
                    img2_array = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2GRAY)
                    
                    # Формуємо матрицю даних
                    data = np.column_stack((img1_array.flatten(), img2_array.flatten()))
                    
                    # Обчислюємо PCA
                    mean = np.mean(data, axis=0)
                    data_adj = data - mean
                    cov = np.cov(data_adj.T)
                    eigenvals, eigenvects = np.linalg.eig(cov)
                    
                    # Проектуємо дані на перший головний компонент
                    merged_array = np.dot(data_adj, eigenvects[:, 0]).reshape(img1_array.shape)
                    merged_array = (merged_array - merged_array.min()) / (merged_array.max() - merged_array.min()) * 255
                    self.merged_image = Image.fromarray(np.uint8(merged_array))
                
                elif method == "Вейвлет перетворення":
                    img1_array = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2GRAY)
                    img2_array = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2GRAY)
                    
                    # Застосовуємо вейвлет-перетворення
                    coeffs1 = pywt.dwt2(img1_array, 'haar')
                    coeffs2 = pywt.dwt2(img2_array, 'haar')
                    
                    # Змішуємо коефіцієнти
                    merged_coeffs = (
                        (coeffs1[0] + coeffs2[0]) / 2,
                        (
                            (coeffs1[1][0] + coeffs2[1][0]) / 2,
                            (coeffs1[1][1] + coeffs2[1][1]) / 2,
                            (coeffs1[1][2] + coeffs2[1][2]) / 2
                        )
                    )
                    
                    # Зворотне перетворення
                    merged_array = pywt.idwt2(merged_coeffs, 'haar')
                    merged_array = np.clip(merged_array, 0, 255)
                    self.merged_image = Image.fromarray(np.uint8(merged_array))
                
                elif method == "Піраміда Лапласа":
                    img1_array = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2GRAY)
                    img2_array = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2GRAY)
                    
                    # Будуємо піраміду Лапласа
                    levels = 4
                    pyramids1 = [img1_array]
                    pyramids2 = [img2_array]
                    
                    for i in range(levels):
                        img1_array = cv2.pyrDown(img1_array)
                        img2_array = cv2.pyrDown(img2_array)
                        pyramids1.append(img1_array)
                        pyramids2.append(img2_array)
                    
                    # Змішуємо рівні піраміди
                    merged_pyramids = []
                    for p1, p2 in zip(pyramids1, pyramids2):
                        merged_pyramids.append((p1 + p2) / 2)
                    
                    # Відновлюємо зображення
                    merged_array = merged_pyramids[-1]
                    for i in range(levels-1, -1, -1):
                        merged_array = cv2.pyrUp(merged_array)
                        if merged_array.shape != merged_pyramids[i].shape:
                            merged_array = cv2.resize(merged_array, (merged_pyramids[i].shape[1], merged_pyramids[i].shape[0]))
                        merged_array = cv2.add(merged_array, merged_pyramids[i])
                    
                    self.merged_image = Image.fromarray(np.uint8(merged_array))
                
                elif method == "Маска прозорості":
                    img1_array = self.to_numpy(self.image1)
                    img2_array = self.to_numpy(self.image2)
                    
                    # Створюємо маску на основі яскравості
                    mask = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
                    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
                    
                    # Застосовуємо маску
                    merged_array = img1_array * mask[:,:,np.newaxis] + img2_array * (1 - mask[:,:,np.newaxis])
                    self.merged_image = self.to_pil(merged_array)
                
                elif method == "Градієнтне злиття":
                    img1_array = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2GRAY)
                    img2_array = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2GRAY)
                    
                    # Обчислюємо градієнти
                    grad_x1 = cv2.Sobel(img1_array, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y1 = cv2.Sobel(img1_array, cv2.CV_64F, 0, 1, ksize=3)
                    grad_x2 = cv2.Sobel(img2_array, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y2 = cv2.Sobel(img2_array, cv2.CV_64F, 0, 1, ksize=3)
                    
                    # Обчислюємо магнітуду градієнтів
                    mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
                    mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
                    
                    # Створюємо маску на основі градієнтів
                    mask = mag1 > mag2
                    
                    # Змішуємо зображення
                    merged_array = np.where(mask, img1_array, img2_array)
                    self.merged_image = Image.fromarray(np.uint8(merged_array))
                
                elif method == "Метод локальної варіації":
                    img1_array = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2GRAY)
                    img2_array = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2GRAY)
                    
                    # Обчислюємо локальну варіацію
                    kernel = np.ones((3,3)) / 9
                    var1 = cv2.filter2D(img1_array**2, -1, kernel) - (cv2.filter2D(img1_array, -1, kernel))**2
                    var2 = cv2.filter2D(img2_array**2, -1, kernel) - (cv2.filter2D(img2_array, -1, kernel))**2
                    
                    # Створюємо маску на основі варіації
                    mask = var1 > var2
                    
                    # Змішуємо зображення
                    merged_array = np.where(mask, img1_array, img2_array)
                    self.merged_image = Image.fromarray(np.uint8(merged_array))

                self.display_image(self.merged_image, self.result_canvas)
            
            except Exception as e:
                messagebox.showerror("Помилка", f"Виникла помилка при злитті зображень: {str(e)}")
        else:
            messagebox.showerror("Помилка", "Будь ласка, завантажте обидва зображення перед злиттям.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMergerApp(root)
    root.mainloop()

