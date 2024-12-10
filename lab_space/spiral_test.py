import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# Hàm tối ưu hóa spiral
def optimized_spiral(x_center, y_center, R, max_points, a=0.1):
    total_area = np.pi * R**2  # Diện tích hình tròn
    avg_spacing = np.sqrt(total_area / max_points)  # Khoảng cách trung bình giữa các điểm
    theta_max = R / avg_spacing * 2 * np.pi  # Góc tối đa cần thiết
    b = (R - a) / theta_max  # Điều chỉnh tham số b để đạt được bán kính R
    
    points = []
    theta_values = np.linspace(0, theta_max, max_points)
    for theta in theta_values:
        r = a + b * theta
        if r > R:
            break
        x = x_center + r * np.cos(theta)
        y = y_center + r * np.sin(theta)
        points.append((x, y))
    return points

# Hàm kiểm tra overlap
def is_overlap(rect1, rect2, max_overlap=0.5):
    x1_min, y1_min = rect1[0] - rect1[2] / 2, rect1[1] - rect1[3] / 2
    x1_max, y1_max = rect1[0] + rect1[2] / 2, rect1[1] + rect1[3] / 2
    x2_min, y2_min = rect2[0] - rect2[2] / 2, rect2[1] - rect2[3] / 2
    x2_max, y2_max = rect2[0] + rect2[2] / 2, rect2[1] + rect2[3] / 2

    overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    overlap_area = overlap_x * overlap_y

    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]
    min_area = min(area1, area2)

    return overlap_area / min_area > max_overlap

# Hàm thêm hình chữ nhật
def add_rectangle(center, width, height, rectangles, max_overlap=0.5):
    new_rect = (center[0], center[1], width, height)
    for rect in rectangles:
        if is_overlap(new_rect, rect, max_overlap):
            return new_rect, False
    rectangles.append(new_rect)
    return new_rect, True

# Tham số
x_center, y_center = 0, 0
R = 1.8
max_points = 200
rect_width = 0.5
rect_height = 0.5

# Tạo danh sách các điểm và hình chữ nhật
spiral_points = optimized_spiral(x_center, y_center, R, max_points)
valid_rectangles, invalid_rectangles = [], []
valid_points, invalid_points = [], []

for point in spiral_points:
    rect, is_valid = add_rectangle(point, rect_width, rect_height, valid_rectangles)
    if is_valid:
        valid_points.append(point)
    else:
        invalid_points.append(point)
        invalid_rectangles.append(rect)

# Chia tách x, y để vẽ
x_valid, y_valid = zip(*valid_points) if valid_points else ([], [])
x_invalid, y_invalid = zip(*invalid_points) if invalid_points else ([], [])

# Tạo hình vẽ
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(left=0.25, right=0.85)

# Vẽ các điểm hợp lệ và không hợp lệ
valid_points_plot, = ax.plot(x_valid, y_valid, 'bo', label='Valid Points')
invalid_points_plot, = ax.plot(x_invalid, y_invalid, 'rx', label='Invalid Points')

# Vẽ hình chữ nhật hợp lệ và không hợp lệ
valid_rectangles_patches = []
for rect in valid_rectangles:
    rect_x = rect[0] - rect[2] / 2
    rect_y = rect[1] - rect[3] / 2
    rectangle = plt.Rectangle((rect_x, rect_y), rect[2], rect[3], color='g', fill=False)
    valid_rectangles_patches.append(rectangle)
    ax.add_patch(rectangle)

invalid_rectangles_patches = []
for rect in invalid_rectangles:
    rect_x = rect[0] - rect[2] / 2
    rect_y = rect[1] - rect[3] / 2
    rectangle = plt.Rectangle((rect_x, rect_y), rect[2], rect[3], color='r', fill=False, linestyle='--')
    invalid_rectangles_patches.append(rectangle)
    ax.add_patch(rectangle)

# Vẽ hình tròn giới hạn
circle = plt.Circle((x_center, y_center), R, color='black', fill=False, linestyle='--', linewidth=1)
ax.add_artist(circle)

# Thêm văn bản số điểm hợp lệ và không hợp lệ
valid_text = f'Valid Points: {len(valid_points)}'
invalid_text = f'Invalid Points: {len(invalid_points)}'
ax.text(0.05, 1.0, valid_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='b')
ax.text(0.05, 0.95, invalid_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', color='r')

# Thêm bảng chú thích
legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

# Tạo CheckButtons để bật/tắt hiển thị
check_ax = plt.axes([0.05, 0.4, 0.15, 0.2])
check = CheckButtons(check_ax, ['Invalid Points', 'Invalid Rectangles'], [True, True])

# Hàm callback để bật/tắt hiển thị
def toggle_visibility(label):
    if label == 'Invalid Points':
        invalid_points_plot.set_visible(not invalid_points_plot.get_visible())
    elif label == 'Invalid Rectangles':
        for patch in invalid_rectangles_patches:
            patch.set_visible(not patch.get_visible())
    fig.canvas.draw()

check.on_clicked(toggle_visibility)

# Cài đặt đồ thị
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f"Spiral with {max_points} Points and Toggleable Invalid Elements")
ax.grid()
ax.axis('equal')
plt.show()
