import pygame
import math
import numpy as np

# Инициализация Pygame
pygame.init()

# Определение цветов
BLACK = (0, 0, 0)# Определение черного цвета
WHITE = (255, 255, 255)# Определение беолго цвета
RED = (255, 0, 0)# Точка зафиксирована - красный цвет
BLUE = (0, 0, 255)# Точка выбрана для построения отрезка 
GREEN = (0, 255, 0)  # Объект выбран в режиме выбора элементов

# Определение размеров окна
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('МиниСАПР')

# Определение параметров системы координат
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
AXIS_LENGTH = min(WIDTH, HEIGHT) // 2 - 20  # Длина осей

# Класс для точек
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fixed = False  # Для фиксации точки на отрезке
        self.color = BLACK

    def draw(self):  # отображение
        color = BLUE if self.fixed else BLACK
        pygame.draw.circle(screen, color, (self.x, self.y), 5)
        font = pygame.font.Font(None, 20)
        text = font.render(f"({self.x}, {self.y})", True, color)
        screen.blit(text, (self.x + 10, self.y - 20))

    def collides(self, other):
        # Проверка на пересечение с другой точкой
        return math.dist((self.x, self.y), (other.x, other.y)) <= 5

    def fix_point(self):  # фиксация точки
        if not self.fixed:
            self.fixed = True
            self.color = BLUE

    def set_coordinates(self, x, y):
        # Устанавливаем новые координаты точки (используется для слияния точек)
        if not self.fixed:
            self.x = x
            self.y = y

        # Если точка зафиксирована, не изменяем её координаты

    # Новый метод для предотвращения изменения координат фиксированной точки
    def update_coordinates(self):
        if self.fixed:
            self.x = self.x
            self.y = self.y


    
# Класс для отрезков
class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.horizontal = False  # Горизонтальность
        self.vertical = False  # Вертикальность
        self.length = 0  # Длина отрезка

    def calculate_properties(self):
        # Рассчитываем свойства отрезка при создании
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        self.length = math.sqrt(dx ** 2 + dy ** 2)
        self.horizontal = dy == 0
        self.vertical = dx == 0

    def draw(self, color=BLACK): # отображение
        pygame.draw.line(screen, color, (self.start.x, self.start.y), (self.end.x, self.end.y), 2)


def fix_point(selected_elements): # фиксация точки
    for element in selected_elements:
        if isinstance(element, Point):
            element.fixed = True  # Устанавливаем флаг фиксации точки

def adjust_distance(point1, point2, desired_distance):
    fixed_points = []  # Ваш список фиксированных точек

    # Если выбраны две точки
    if isinstance(point1, Point) and isinstance(point2, Point):
        if point1 in fixed_points and point2 in fixed_points:
            return

        vector = np.array([point2.x - point1.x, point2.y - point1.y])
        current_distance = np.linalg.norm(vector)

        if current_distance == 0:  
            return

        ratio = desired_distance / current_distance
        new_vector = vector * ratio

        if point1 not in fixed_points and not point2 in fixed_points:
            # Обновляем координаты обеих точек, если ни одна из них не фиксирована
            point1.x, point1.y = round(point2.x - new_vector[0]), round(point2.y - new_vector[1])
            point2.x, point2.y = round(point1.x + new_vector[0]), round(point1.y + new_vector[1])
        elif point1 not in fixed_points:
            # Обновляем координаты только первой точки, если вторая зафиксирована
            new_x1, new_y1 = round(point2.x - new_vector[0]), round(point2.y - new_vector[1])
            point1.x, point1.y = new_x1, new_y1
        elif point2 not in fixed_points:
            # Обновляем координаты только второй точки, если первая зафиксирована
            new_x2, new_y2 = round(point1.x + new_vector[0]), round(point1.y + new_vector[1])
            point2.x, point2.y = new_x2, new_y2
    
    # Если выбран отрезок и две точки
    elif isinstance(point1, Segment) and isinstance(point2, Point):
        segment = point1
        point1 = segment.start
        point2 = segment.end

        if point1 in fixed_points and point2 in fixed_points:
            return

        vector = np.array([point2.x - point1.x, point2.y - point1.y])
        current_distance = np.linalg.norm(vector)

        if current_distance == 0:  
            return

        ratio = desired_distance / current_distance
        new_vector = vector * ratio

        if point1 not in fixed_points and not point2 in fixed_points:
            # Обновляем координаты обеих точек, если ни одна из них не фиксирована
            segment.start.x, segment.start.y = round(point2.x - new_vector[0]), round(point2.y - new_vector[1])
            segment.end.x, segment.end.y = round(point1.x + new_vector[0]), round(point1.y + new_vector[1])
        elif point1 not in fixed_points:
            # Обновляем координаты только первой точки, если вторая зафиксирована
            new_x1, new_y1 = round(point2.x - new_vector[0]), round(point2.y - new_vector[1])
            segment.start.x, segment.start.y = new_x1, new_y1
        elif point2 not in fixed_points:
            # Обновляем координаты только второй точки, если первая зафиксирована
            new_x2, new_y2 = round(point1.x + new_vector[0]), round(point1.y + new_vector[1])
            segment.end.x, segment.end.y = new_x2, new_y2

def make_parallel(segment1, segment2):
    if all(elem.fixed for elem in [segment1.start, segment1.end, segment2.start, segment2.end]):
        return

    vec1 = np.array([segment1.end.x - segment1.start.x, segment1.end.y - segment1.start.y])
    vec2 = np.array([segment2.end.x - segment2.start.x, segment2.end.y - segment2.start.y])

    len1 = np.linalg.norm(vec1)
    len2 = np.linalg.norm(vec2)

    if len1 == 0 or len2 == 0:
        # При нулевой длине вектора - произведите перестройку без учета ограничений
        if not segment2.start.fixed and not segment2.end.fixed:
            segment2.end.x = segment2.start.x + vec1[1]
            segment2.end.y = segment2.start.y - vec1[0]
        elif not segment1.start.fixed and not segment1.end.fixed:
            segment1.start.x = segment2.start.x - vec2[1]
            segment1.start.y = segment2.start.y + vec2[0]
            segment1.end.x = segment1.start.x + (segment1.end.x - segment1.start.x)
            segment1.end.y = segment1.start.y + (segment1.end.y - segment1.start.y)
    else:
        normalized_vec1 = vec1 / len1
        normalized_vec2 = vec2 / len2

        projection = np.dot(vec2, normalized_vec1)
        new_vec2 = normalized_vec1 * projection

        if not segment2.start.fixed and not segment2.end.fixed:
            length_segment2 = np.linalg.norm(vec2)
            vec2_normalized = new_vec2 / np.linalg.norm(new_vec2)
            print(vec2_normalized[0], vec2_normalized[1], new_vec2, np.linalg.norm(new_vec2))
            segment2.end.x = segment2.start.x + int(vec2_normalized[0] * length_segment2)
            segment2.end.y = segment2.start.y + int(vec2_normalized[1] * length_segment2)
        elif not segment1.start.fixed and not segment1.end.fixed:
            projection = np.dot(vec1, normalized_vec2)
            new_vec1 = normalized_vec2 * projection

            segment1.start.x = segment2.start.x - int(new_vec1[0])
            segment1.start.y = segment2.start.y - int(new_vec1[1])
            segment1.end.x = segment1.start.x + (segment1.end.x - segment1.start.x)
            segment1.end.y = segment1.start.y + (segment1.end.y - segment1.start.y)

def make_perpendicular(segment1, segment2):
    # Проверяем, что точки отрезков не зафиксированы
    if all(not point.fixed for point in [segment1.start, segment1.end, segment2.start, segment2.end]):
        # Находим центр отрезка AB
        center_AB = ((segment1.start.x + segment1.end.x) / 2, (segment1.start.y + segment1.end.y) / 2)
        
        # Находим вектор AB
        vector_AB = (segment1.end.x - segment1.start.x, segment1.end.y - segment1.start.y)
        
        # Находим вектор, перпендикулярный AB
        perpendicular_vector_AB = (-vector_AB[1], vector_AB[0])
        
        # Находим координаты двух новых точек, добавляя вектор к центру отрезка AB
        new_point1 = (center_AB[0] + perpendicular_vector_AB[0], center_AB[1] + perpendicular_vector_AB[1])
        new_point2 = (center_AB[0] - perpendicular_vector_AB[0], center_AB[1] - perpendicular_vector_AB[1])
        
        # Перемещаем точки segment2 так, чтобы они стали перпендикулярными к отрезку segment1
        length = math.sqrt((segment2.end.x - segment2.start.x) ** 2 + (segment2.end.y - segment2.start.y) ** 2)
        half_length = length / 2
        
        segment2.start.x = center_AB[0] + perpendicular_vector_AB[0] * half_length / length
        segment2.start.y = center_AB[1] + perpendicular_vector_AB[1] * half_length / length
        
        segment2.end.x = center_AB[0] - perpendicular_vector_AB[0] * half_length / length
        segment2.end.y = center_AB[1] - perpendicular_vector_AB[1] * half_length / length

def adjust_angle(segment1, segment2, desired_angle):
    # Функция для вычисления угла между векторами
    def calculate_angle(vec1, vec2):
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
        cos_theta = dot_product / (magnitude1 * magnitude2)
        return math.acos(cos_theta)

    # Проверяем фиксацию точек на отрезках
    if all(point.fixed for point in [segment1.start, segment1.end, segment2.start, segment2.end]):
        return  # Если все точки фиксированы, выходим из функции

    # Получаем векторы отрезков
    vec1 = [segment1.end.x - segment1.start.x, segment1.end.y - segment1.start.y]
    vec2 = [segment2.end.x - segment2.start.x, segment2.end.y - segment2.start.y]

    # Вычисляем угол между отрезками
    angle_rad = calculate_angle(vec1, vec2)

    # Вычисляем разницу между текущим и желаемым углом
    angle_diff = math.radians(desired_angle) - angle_rad

    # Если первый отрезок имеет хотя бы одну неподвижную точку, поворачиваем второй отрезок
    if segment1.start.fixed or segment1.end.fixed:
        new_x = vec2[0] * math.cos(angle_diff) - vec2[1] * math.sin(angle_diff)
        new_y = vec2[0] * math.sin(angle_diff) + vec2[1] * math.cos(angle_diff)
        segment2.end.x = segment2.start.x + new_x
        segment2.end.y = segment2.start.y + new_y
    # Если второй отрезок имеет хотя бы одну неподвижную точку, поворачиваем первый отрезок
    elif segment2.start.fixed or segment2.end.fixed:
        new_x = vec1[0] * math.cos(-angle_diff) - vec1[1] * math.sin(-angle_diff)
        new_y = vec1[0] * math.sin(-angle_diff) + vec1[1] * math.cos(-angle_diff)
        segment1.end.x = segment1.start.x + new_x
        segment1.end.y = segment1.start.y + new_y
    else:
        # Общая точка для вращения отрезков
         central_point = segment1.start

    # Векторы от центральной точки до конечных точек отрезков
         vec1_central = [segment1.end.x - central_point.x, segment1.end.y - central_point.y]
         vec2_central = [segment2.start.x - central_point.x, segment2.start.y - central_point.y]

    # Вращаем обе конечные точки каждого отрезка
         new_x1 = vec1_central[0] * math.cos(angle_diff) - vec1_central[1] * math.sin(angle_diff)
         new_y1 = vec1_central[0] * math.sin(angle_diff) + vec1_central[1] * math.cos(angle_diff)
         new_x2 = vec2_central[0] * math.cos(angle_diff) - vec2_central[1] * math.sin(angle_diff)
         new_y2 = vec2_central[0] * math.sin(angle_diff) + vec2_central[1] * math.cos(angle_diff)

    # Обновляем координаты конечных точек отрезков
         segment1.end.x = central_point.x + new_x1
         segment1.end.y = central_point.y + new_y1

         segment2.start.x = central_point.x + new_x2
         segment2.start.y = central_point.y + new_y2

def adjust_angle_triangle(point, segment1, segment2, segment3, desired_angle):
    # Функция для вычисления угла между векторами
    def calculate_angle(vec1, vec2):
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
        cos_theta = dot_product / (magnitude1 * magnitude2)
        return math.acos(cos_theta)

    # Проверяем фиксацию точек и отрезков
    if all(point.fixed for point in [point, segment1.start, segment1.end, segment2.start, segment2.end, segment3.start, segment3.end]):
        return  # Если все точки и отрезки фиксированы, выходим из функции

    # Получаем векторы отрезков от выбранной точки
    vec1 = [segment1.end.x - point.x, segment1.end.y - point.y]
    vec2 = [segment2.start.x - point.x, segment2.start.y - point.y]
    vec3 = [segment3.start.x - point.x, segment3.start.y - point.y]

    # Вычисляем углы между векторами
    angle1 = calculate_angle(vec1, vec2)
    angle2 = calculate_angle(vec2, vec3)
    angle3 = calculate_angle(vec3, vec1)

    # Сумма углов в треугольнике должна быть 180 градусов
    current_sum_angles = math.degrees(angle1 + angle2 + angle3)
    angle_diff = math.radians(desired_angle) - current_sum_angles

    # Вращаем второй отрезок
    new_x2 = vec2[0] * math.cos(angle_diff) - vec2[1] * math.sin(angle_diff)
    new_y2 = vec2[0] * math.sin(angle_diff) + vec2[1] * math.cos(angle_diff)
    segment2.start.x = point.x + new_x2
    segment2.start.y = point.y + new_y2

    # Вращаем третий отрезок
    new_x3 = vec3[0] * math.cos(angle_diff) - vec3[1] * math.sin(angle_diff)
    new_y3 = vec3[0] * math.sin(angle_diff) + vec3[1] * math.cos(angle_diff)
    segment3.start.x = point.x + new_x3
    segment3.start.y = point.y + new_y3

def make_horizontal(segment): # сделать отрезок горизонтальным
    # Сохраняем начальные координаты
    start_x, start_y = segment.start.x, segment.start.y
    end_x, end_y = segment.end.x, segment.end.y

    # Проверяем, являются ли начальная и конечная точки зафиксированными
    start_fixed = segment.start.fixed
    end_fixed = segment.end.fixed

    # Если обе точки зафиксированы, не изменяем их координаты
    if start_fixed and end_fixed:
        return

    # Считаем текущую длину отрезка
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx ** 2 + dy ** 2)
    
    # Устанавливаем координаты конечной точки для горизонтального отрезка, сохраняя изначальную длину
    segment.end.x = start_x + length
    segment.end.y = start_y

def make_vertical(segment): # сделать отрезок вертикальным
    # Сохраняем начальные координаты
    start_x, start_y = segment.start.x, segment.start.y
    end_x, end_y = segment.end.x, segment.end.y
    
    # Проверяем, являются ли начальная и конечная точки зафиксированными
    start_fixed = segment.start.fixed
    end_fixed = segment.end.fixed

    # Если обе точки зафиксированы, не изменяем их координаты
    if start_fixed and end_fixed:
        return

    # Считаем текущую длину отрезка
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx ** 2 + dy ** 2)
    
    # Устанавливаем координаты конечной точки для вертикального отрезка, сохраняя изначальную длину
    segment.end.x = start_x
    segment.end.y = start_y + length


# Метод для задания расстояния между двумя точками
def make_distance(selected_elements): 
    if len(selected_elements) == 2 and all(isinstance(element, Point) for element in selected_elements):
        distance = float(input("Введите расстояние между точками: "))
        point1, point2 = selected_elements
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        current_distance = math.sqrt(dx ** 2 + dy ** 2)
        ratio = distance / current_distance if current_distance != 0 else 0
        # Изменяем положение второй точки, чтобы расстояние было равно введенному пользователем
        point2.x = point1.x + int(dx * ratio)
        point2.y = point1.y + int(dy * ratio)

def execute_action(selected_elements, current_key):
    # В этой функции можно реализовать применение ограничений для выбранных элементов
    if current_key == pygame.K_1:
        fix_point(selected_elements)
    elif current_key == pygame.K_2:
        make_distance(selected_elements)
    # Другие ограничения (2-9) могут быть обработаны здесь
    else:
        print("Нет обработчика для данной клавиши")

def get_input(screen, prompt):
    font = pygame.font.Font(None, 36)
    input_box = pygame.Rect(100, 100, 140, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = ''
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        return text
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
        screen.fill((30, 30, 30))
        txt_surface = font.render(prompt + text, True, color)
        width = max(200, txt_surface.get_width() + 10)
        input_box.w = width
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        pygame.draw.rect(screen, color, input_box, 2)
        pygame.display.flip()

def merge_points(point1, point2):
    global points, segments

    # Проверка на то, является ли point1 началом или концом отрезка
    connected_segments = [segment for segment in segments if segment.start == point1 or segment.end == point1]

    if not connected_segments:
        # Если point1 не является началом или концом отрезка
        # Заменяем координаты point1 на координаты point2
        point1.x, point1.y = point2.x, point2.y

        # Удаляем point2 из списка точек
        points.remove(point2)

        # Обновляем свойства отрезков
        for segment in segments:
            if segment.start == point2:
                segment.start = point1
            if segment.end == point2:
                segment.end = point1

    else:
        # Если point1 является началом или концом отрезка, то сдвигаем только point1
        dx = point2.x - point1.x
        dy = point2.y - point1.y

        # Сдвигаем координаты point1 на вектор (dx, dy)
        point1.x += dx
        point1.y += dy

        # Удаляем point2 из списка точек
        points.remove(point2)

        # Обновляем свойства отрезков
        for segment in segments:
            if segment.start == point2:
                segment.start = point1
            if segment.end == point2:
                segment.end = point1

def point_belongs_to_segment(point1, point2, segment, moving_point, distance_from_end):
    # Параметр t для вычисления точки на отрезке
    t = 1 - distance_from_end / math.dist((point1.x, point1.y), (point2.x, point2.y))

    # Вычислите координаты точки на отрезке
    x_on_line = point1.x + t * (point2.x - point1.x)
    y_on_line = point1.y + t * (point2.y - point1.y)

    # Переместите выбранную точку на расстояние от конца отрезка
    moving_point.x, moving_point.y = x_on_line, y_on_line

    # Обновите координаты точек отрезка, если они фиксированы
    if segment.start.fixed:
        segment.start.x, segment.start.y = point1.x, point1.y
    if segment.end.fixed:
        segment.end.x, segment.end.y = point2.x, point2.y

# Ваш код остается таким же, но добавим функции

# В вашем цикле обработки событий

        
def display_selected_elements():
    print("Выбранные элементы:")
    for element in selected_elements:
        print(element)

key_bindings = {
    pygame.K_1: fix_point,
    pygame.K_2: adjust_distance,
    pygame.K_3: make_parallel,
    pygame.K_4: make_perpendicular,
    pygame.K_5: adjust_angle,
    pygame.K_6: make_horizontal,
    pygame.K_7: make_vertical,
    pygame.K_8: point_belongs_to_segment,
    pygame.K_9: point_belongs_to_segment
    # ... добавьте остальные функции
}
# Списки для хранения точек и отрезков
points = []
segments = []
running = True
creating_segment = False  # Переменная для отслеживания создания отрезка
selected_points = []  # Переменная для отслеживания создания отрезка
select_mode = False # Переменная для отслеживания режима выбора элементов
selected_elements = []
fixed_points = []
selected_points_creation = []  # Список выбранных точек для создания отрезков
selected_points_selection = []  # Список выбранных точек в режиме выбора элементов
points_in_segment = []  # Объявление переменной перед использованием


while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if not select_mode:
                if event.button == 1: # создание точки
                    new_point = Point(x, y)
                    points.append(new_point)
                elif event.button == 3: # создание отрезка
                    for point in points:
                        if math.dist((point.x, point.y), (x, y)) <= 5:
                            if len(selected_points_creation) == 0:
                                selected_points_creation.append(point)
                                point.fixed = True
                            elif len(selected_points_creation) == 1:
                                new_segment = Segment(selected_points_creation[0], point)
                                new_segment.calculate_properties()
                                segments.append(new_segment)
                                selected_points_creation[0].fixed = False
                                selected_points_creation.clear()
                                selected_elements.append(new_segment)
            else:  # удаление объектов
                for point in points:
                  if math.dist((point.x, point.y), (x, y)) <= 5:
                     if point not in selected_elements:
                        selected_elements.append(point)
                for segment in segments:
                     if (segment.start in selected_elements and segment.end in selected_elements) or \
                          (segment.end in selected_elements and segment.start in selected_elements):
                       if segment not in selected_elements:
                          selected_elements.append(segment)
                     
        elif event.type == pygame.KEYDOWN: # вход в режим выбора элементов для ограничения
            if event.key == pygame.K_SPACE:
                select_mode = not select_mode
                print("Клавиша пробел нажата") 
                if select_mode:
                    print("Режим выбора элементов включен")
                    selected_elements = []
                else:
                    print("Режим выбора элементов выключен")
                    selected_elements = []
            elif event.key in key_bindings and not select_mode:
                current_key = event.key
                execute_action(selected_elements, current_key)
                selected_elements = []
            
            elif event.key == pygame.K_0 and select_mode:
                display_selected_elements()
                fixed_points = []
            
            elif event.key == pygame.K_p:  # Вывод зафиксированных точек в консоль по нажатию на кнопку P
                for point in fixed_points:
                 print(f"Зафиксированная точка: ({round(point.x)}, {round(point.y)})")

            elif event.key == pygame.K_1 and select_mode:  # фиксация точки
                selected_points = [element for element in selected_elements if isinstance(element, Point)]
                print(selected_points)
                if len(selected_points) == 1:
                  selected_points[0].fix_point()  # Фиксируем выбранную точку
                else:
                  print("Выберите одну точку для выполнения операции")

            elif event.key == pygame.K_2 and select_mode:  # Расстояние между точками
                  if len(selected_elements) == 2 and all(isinstance(elem, Point) for elem in selected_elements):
                         distance = int(input("Введите желаемое расстояние: "))
                         point1, point2 = selected_elements[0], selected_elements[1]
                         adjust_distance(point1, point2, distance)
                  elif len(selected_elements) == 3 and isinstance(selected_elements[2], Segment):
                         segment = selected_elements[2]
                         point1, point2 = selected_elements[0], selected_elements[1]
                         if (point1 == segment.start and point2 == segment.end) or (point2 == segment.start and point1 == segment.end):
                             distance = int(input("Введите желаемое расстояние: "))
                             adjust_distance(point1, point2, distance)
                  else:
                         print("Выберите 2 точки для выполнения операции")

            elif event.key == pygame.K_3 and select_mode:  
                selected_points = [element for element in selected_elements if isinstance(element, Point)]
                selected_segments = [element for element in selected_elements if isinstance(element, Segment)]

                if len(selected_points) == 4 and len(selected_segments) == 2:
                  segment1, segment2 = selected_segments
                  make_parallel(segment1, segment2)  
                else:
                  print("Выберите 4 точки и 2 отрезка для выполнения операции")

            elif event.key == pygame.K_4 and select_mode:  
                    selected_points = [element for element in selected_elements if isinstance(element, Point)]
                    selected_segments = [element for element in selected_elements if isinstance(element, Segment)]

                    if len(selected_points) == 4 and len(selected_segments) == 2:
                        segment1, segment2 = selected_segments
                        make_perpendicular(segment1, segment2)
                    else:
                        print("Выберите 4 точки и 2 отрезка для выполнения операции")
            
            elif event.key == pygame.K_5 and select_mode:
                selected_segments = [elem for elem in selected_elements if isinstance(elem, Segment)]
                selected_points = [elem for elem in selected_elements if isinstance(elem, Point)]
                if len(selected_segments) == 2 and len(selected_points) == 3:
                   angle_input = get_input(screen, "Введите угол: ")
                   if angle_input:
                     angle = float(angle_input)
                     adjust_angle(selected_segments[0], selected_segments[1], angle)
                else:
                     print("Выберите два отрезка и три точки для выполнения операции.")
           
            elif event.key == pygame.K_6 and select_mode:  # Сделать отрезок горизонтальным
                
                selected_segments = [element for element in selected_elements if isinstance(element, Segment)]
                print(selected_segments)
                if len(selected_segments) == 1:
                    make_horizontal(selected_segments[0])  # Применяем операцию к выбранному отрезку
                else:
                   print("Выберите один отрезок для выполнения операции")


            elif event.key == pygame.K_7 and select_mode:  # Сделать отрезок вертикальным
               selected_segments = [element for element in selected_elements if isinstance(element, Segment)]
               print(selected_segments)
               if len(selected_segments) == 1:
                    make_vertical(selected_segments[0])  # Применяем операцию к выбранному отрезку
               else:
                   print("Выберите один отрезок для выполнения операции")

            elif event.key == pygame.K_8 and select_mode:
               selected_points = [element for element in selected_elements if isinstance(element, Point)]
               selected_segments = [element for element in selected_elements if isinstance(element, Segment)]

               if len(selected_points) == 3 and len(selected_segments) == 1:
                segment = selected_segments[0]
                point_belongs_to_segment(selected_points[0], selected_points[1], segment, selected_points[2], 200)
                selected_elements = []  # Сброс выбранных элементов после проверки
            
            elif event.key == pygame.K_q and len(selected_elements) == 2 and all(isinstance(elem, Point) for elem in selected_elements):
                merge_points(selected_elements[0], selected_elements[1])
                selected_elements = []  # Очищаем выбранные элементы после слияния
           
       
            elif event.key == pygame.K_r and not select_mode:  # Добавлена проверка на клавишу R и режим не выбора элементов
            # Проверяем, есть ли элемент под курсором мыши для удаления
              x, y = pygame.mouse.get_pos()
              element_found = False
              if not select_mode:
                for point in points:
                 if math.dist((point.x, point.y), (x, y)) <= 5:
                    points.remove(point)
                    element_found = True
                    if point in fixed_points:
                      fixed_points.remove(point)
                    break
                 if not element_found:
                # Проверяем отрезки
                  for segment in segments:                   
                    if (x >= min(segment.start.x, segment.end.x) and x <= max(segment.start.x, segment.end.x) and
                            y >= min(segment.start.y, segment.end.y) and y <= max(segment.start.y, segment.end.y)):
                        segments.remove(segment)
                        element_found = True
                        break
              if not element_found:
                 print("Нет элементов для удаления.")

    for point in fixed_points:
          pygame.draw.circle(screen, RED, (point.x, point.y), 5)
          font = pygame.font.Font(None, 20)
          text = font.render(f"({round(point.x)}, {round(point.y)})", True, BLACK)
          screen.blit(text, (point.x + 10, point.y - 20))



    for point in points: # Цвет точек в зависимости от условий
        if point in selected_points:
           color = RED
        elif point.fixed:
           color = BLUE
        elif point in selected_elements:
           color = GREEN
        else:
           color = BLACK

        pygame.draw.circle(screen, color, (point.x, point.y), 5)
        font = pygame.font.Font(None, 20)
        text = font.render(f"({round(point.x)}, {round(point.y)})", True, BLACK)
        screen.blit(text, (point.x + 10, point.y - 20))

    for segment in segments:
        # Определяем, являются ли начальная и конечная точки отрезка выбранными для наложения ограничений
        start_selected = segment.start in selected_elements
        end_selected = segment.end in selected_elements

        if start_selected and end_selected:
            segment.draw(GREEN)  # Отрисовываем зеленым цветом отрезки между выбранными точками
        else:
            segment.draw() 

    pygame.display.flip()

pygame.quit()