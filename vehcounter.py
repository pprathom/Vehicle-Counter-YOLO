import cv2
import argparse
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def ccw(A, B, C):
    """
    เช็คการจัดเรียงจุดทวนเข็มนาฬิกา เพื่อใช้หาการตัดกันของเส้น
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """
    ตรวจว่าสัดส่วนของเส้นตรง AB ตัดกับ CD หรือไม่
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# ตัวแปร Global สำหรับรองรับการวาดเส้น
line_pts = []
current_mouse_pos = None
stop_counting = False
manual_reload = False

def draw_line(event, x, y, flags, param):
    """
    Callback function สำหรับรับค่าการคลิกเมาส์ 2 จุดบน OpenCV Window
    และติดตามตำแหน่งเมาส์เพื่อวาดเส้นนำสายตา
    """
    global line_pts, current_mouse_pos

    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        # ถ้าคลิกครบ 2 จุดแล้วให้เริ่มใหม่
        if len(line_pts) == 2:
            line_pts = []
        line_pts.append((x, y))

def click_stop_button(event, x, y, flags, param):
    """
    Callback function สำหรับตรวจสอบการคลิกปุ่ม Stop & Export และ Reload บนหน้าจอ
    """
    global stop_counting, manual_reload
    if event == cv2.EVENT_LBUTTONDOWN:
        btn_x1, btn_y1, btn_x2, btn_y2, rld_x1, rld_y1, rld_x2, rld_y2 = param

        # เช็คปุ่ม Stop
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            stop_counting = True

        # เช็คปุ่ม Reload
        if rld_x1 <= x <= rld_x2 and rld_y1 <= y <= rld_y2:
            manual_reload = True

# =========================================
# ฟังก์ชันช่วยวาด UI บน OpenCV
# =========================================

def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """วาด Rectangle มุมมน"""
    x1, y1 = pt1
    x2, y2 = pt2
    # Fill main rectangles
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    # Fill corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_text_centered(img, text, center_x, y, font, scale, color, thickness=1):
    """วาดข้อความ align center"""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (center_x - tw // 2, y), font, scale, color, thickness, cv2.LINE_AA)

def run_counter(source, export_csv, fast_mode):
    """
    ฟังก์ชันหลักในการนับยานพาหนะเมื่อได้ Source และ Export Path แล้ว
    """
    global line_pts, stop_counting

    print(f"กำลังเปิดโหลดโมเดลและ Source: {source}")
    model = YOLO("yolov8n.pt")

    vehicle_classes = [2, 3, 5, 7]
    class_names = model.names

    track_history = defaultdict(list)
    counted_ids = set()

    counts = {
        "inbound": defaultdict(int),
        "outbound": defaultdict(int)
    }

    export_data = []

    # ฟังก์ชันช่วยโหลด VideoCapture เพื่อให้เรียกซ้ำตอน reload ได้
    def connect_stream():
        nonlocal cap
        if 'cap' in locals() and cap is not None:
            cap.release()

        cap_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    cap = None
    cap = connect_stream()

    if not cap.isOpened():
        print(f"Error: ไม่สามารถเปิด video/stream '{source}' ได้")
        messagebox.showerror("Error", f"ไม่สามารถเปิด Video หรือ Streaming นี้ได้: {source}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: ไม่สามารถอ่านเฟรมแรกจาก video ได้")
        messagebox.showerror("Error", "อ่านเฟรมแรกของวิดีโอไม่ได้")
        cap.release()
        return

    h, w = frame.shape[:2]
    max_height = 480
    scale = 1.0
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = w, h

    frame = cv2.resize(frame, (new_w, new_h))

    # ==========================================
    # ส่วนที่ 1: UI สำหรับคลิกวาดเส้น (Setup Line) — อัปเกรด UI
    # ==========================================
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    cv2.namedWindow("Setup Line")
    cv2.setMouseCallback("Setup Line", draw_line)

    print("===== กำลังตั้งค่าเส้นสมมติ =====")

    while True:
        temp_frame = frame.copy()

        # วาดเส้นถ้ามีการคลิกจุด
        if len(line_pts) >= 1:
            cv2.circle(temp_frame, line_pts[0], 7, (0, 220, 255), -1)
            cv2.circle(temp_frame, line_pts[0], 9, (255, 255, 255), 1)

        if len(line_pts) == 1 and current_mouse_pos is not None:
            cv2.line(temp_frame, line_pts[0], current_mouse_pos, (0, 220, 255), 1, cv2.LINE_AA)

        if len(line_pts) == 2:
            cv2.line(temp_frame, line_pts[0], line_pts[1], (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(temp_frame, line_pts[1], 7, (0, 220, 255), -1)
            cv2.circle(temp_frame, line_pts[1], 9, (255, 255, 255), 1)
            cv2.putText(temp_frame, "Counting Line", (line_pts[0][0] + 5, line_pts[0][1] - 12),
                        FONT, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        # แถบคำแนะนำที่ด้านล่าง (สวยงามกว่าเดิม)
        bar_h = 45
        overlay = temp_frame.copy()
        cv2.rectangle(overlay, (0, new_h - bar_h), (new_w, new_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, temp_frame, 0.25, 0, temp_frame)

        hint = "Click 2 points to draw a counting line  |  [C] Confirm  |  [Q] Quit"
        draw_text_centered(temp_frame, hint, new_w // 2, new_h - 13, FONT, 0.45, (200, 200, 200), 1)

        # Header bar
        overlay2 = temp_frame.copy()
        cv2.rectangle(overlay2, (0, 0), (new_w, 38), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.8, temp_frame, 0.2, 0, temp_frame)
        draw_text_centered(temp_frame, "SETUP  COUNTING  LINE", new_w // 2, 25, FONT, 0.65, (0, 220, 255), 2)

        cv2.imshow("Setup Line", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if len(line_pts) == 2:
                line_start, line_end = line_pts[0], line_pts[1]
                break
            else:
                print("กรุณาคลิกให้ครบ 2 จุดก่อนกด 'c' เพื่อยืนยัน")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Setup Line")

    # ==========================================
    # ส่วนที่ 2: เริ่มตรวจจับยานพาหนะและนับจำนวน — Dashboard อัปเกรด
    # ==========================================
    cv2.namedWindow("Vehicle Counting")
    cv2.namedWindow("Dashboard")

    # Dashboard canvas
    dash_w, dash_h = 380, 520

    # ปุ่ม STOP & EXPORT
    btn_w, btn_h = 160, 44
    btn_x1 = (dash_w - btn_w) // 2
    btn_y1 = dash_h - btn_h - 20
    btn_x2 = btn_x1 + btn_w
    btn_y2 = btn_y1 + btn_h

    # ปุ่ม RELOAD
    rld_w, rld_h = 160, 44
    rld_x1 = btn_x1
    rld_y1 = btn_y1 - rld_h - 12
    rld_x2 = rld_x1 + rld_w
    rld_y2 = rld_y1 + rld_h

    cv2.setMouseCallback("Dashboard", click_stop_button,
                         param=(btn_x1, btn_y1, btn_x2, btn_y2, rld_x1, rld_y1, rld_x2, rld_y2))

    global manual_reload
    last_frame_time = datetime.now()

    # --- สี Palette ---
    BG_COLOR      = (18, 18, 28)        # พื้นหลัง dark navy
    HEADER_COLOR  = (28, 28, 45)        # Header bar
    ACCENT        = (0, 200, 255)       # Cyan accent
    INBOUND_COLOR = (80, 220, 130)      # เขียว
    OUTBOUND_COLOR= (80, 130, 255)      # น้ำเงิน
    TEXT_COLOR    = (220, 220, 230)     # ข้อความทั่วไป
    MUTED_COLOR   = (110, 110, 130)     # ข้อความเบา
    BTN_STOP      = (40, 50, 200)       # ปุ่ม Stop (indigo)
    BTN_RELOAD    = (20, 140, 200)      # ปุ่ม Reload (teal)
    DIVIDER_COLOR = (40, 40, 60)        # เส้นคั่น

    # Vehicle class icons
    cls_icon = {
        "car":        "🚗",
        "motorcycle": "🏍",
        "bus":        "🚌",
        "truck":      "🚚",
    }

    while True:
        if stop_counting:
            break

        if not cap.isOpened():
            print("Stream หลุด! กำลังพยายามเชื่อมต่อใหม่...")
            cap = connect_stream()
            if not cap.isOpened():
                cv2.waitKey(1000)
                continue

        ret, frame = cap.read()

        time_since_last_frame = (datetime.now() - last_frame_time).total_seconds()
        is_stalled = (not ret and not str(source).isdigit()) or \
                     (time_since_last_frame > 5.0 and not str(source).isdigit())

        if is_stalled or manual_reload:
            if manual_reload:
                print("ผู้ใช้กดปุ่ม Reload! กำลังโหลด Stream ใหม่...")
                manual_reload = False
            else:
                print("สตรีมมิ่งค้าง... กำลังโหลดใหม่ (Auto-Reload)")

            cap = connect_stream()
            last_frame_time = datetime.now()
            continue

        if not ret:
            if str(source).isdigit() or source.endswith('.mp4') or source.endswith('.avi'):
                print("สิ้นสุดไฟล์วิดีโอ")
                break
            continue

        last_frame_time = datetime.now()
        frame = cv2.resize(frame, (new_w, new_h))

        tracking_imgsz = 320 if fast_mode else 640
        results = model.track(frame, persist=True, classes=vehicle_classes,
                               imgsz=tracking_imgsz, verbose=False)

        # วาดเส้นนับ (gradient look)
        cv2.line(frame, line_start, line_end, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, line_start, 5, (255, 255, 255), -1)
        cv2.circle(frame, line_end,   5, (255, 255, 255), -1)
        cv2.putText(frame, "COUNTING LINE", (line_start[0] + 6, line_start[1] - 10),
                    FONT, 0.4, (0, 220, 255), 1, cv2.LINE_AA)

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                cls_name = class_names[class_id]
                label = f"{cls_name}  #{track_id}"

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 1)

                # Label background
                (lw, lh), _ = cv2.getTextSize(label, FONT, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (0, 220, 255), -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4),
                            FONT, 0.4, (10, 10, 10), 1, cv2.LINE_AA)

                # Trajectory trail (fade effect)
                hist = track_history[track_id]
                for i in range(1, len(hist)):
                    alpha = int(255 * i / len(hist))
                    color = (alpha // 2, alpha, 200)
                    cv2.line(frame, hist[i - 1], hist[i], color, 1, cv2.LINE_AA)

                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

                if len(track_history[track_id]) >= 2:
                    prev_pt = track_history[track_id][-2]
                    curr_pt = track_history[track_id][-1]

                    if track_id not in counted_ids:
                        if intersect(line_start, line_end, prev_pt, curr_pt):
                            counted_ids.add(track_id)
                            is_inbound = ccw(line_start, line_end, curr_pt)
                            direction = "Inbound" if is_inbound else "Outbound"

                            if is_inbound:
                                counts["inbound"][cls_name] += 1
                            else:
                                counts["outbound"][cls_name] += 1

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            export_data.append([timestamp, track_id, cls_name, direction])

        # ====================
        # วาด Dashboard ใหม่
        # ====================
        dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
        dashboard[:] = BG_COLOR

        # --- Header ---
        cv2.rectangle(dashboard, (0, 0), (dash_w, 58), HEADER_COLOR, -1)
        draw_text_centered(dashboard, "VEHICLE COUNTER", dash_w // 2, 24,
                           FONT, 0.65, ACCENT, 2)
        now_str = datetime.now().strftime("%H:%M:%S")
        draw_text_centered(dashboard, now_str, dash_w // 2, 46,
                           FONT, 0.42, MUTED_COLOR, 1)

        # --- ยอดนับรวม ---
        total_in  = sum(counts["inbound"].values())
        total_out = sum(counts["outbound"].values())
        total_all = total_in + total_out

        # Total box
        cv2.rectangle(dashboard, (15, 68), (dash_w - 15, 108), DIVIDER_COLOR, -1)
        draw_text_centered(dashboard, f"TOTAL  {total_all}", dash_w // 2, 95,
                           FONT, 0.75, (255, 255, 255), 2)

        # Divider
        cv2.line(dashboard, (15, 116), (dash_w - 15, 116), DIVIDER_COLOR, 1)

        # --- INBOUND section ---
        y = 138
        cv2.rectangle(dashboard, (15, y - 16), (130, y + 6), (0, 120, 60), -1)
        cv2.putText(dashboard, "  INBOUND", (15, y),
                    FONT, 0.5, INBOUND_COLOR, 1, cv2.LINE_AA)

        total_in_str = f"{total_in}"
        (tw, _), _ = cv2.getTextSize(total_in_str, FONT, 0.6, 2)
        cv2.putText(dashboard, total_in_str, (dash_w - 25 - tw, y),
                    FONT, 0.6, INBOUND_COLOR, 2, cv2.LINE_AA)

        y += 14
        for cls_name, count in counts["inbound"].items():
            y += 26
            icon_name = cls_name.lower()
            cv2.putText(dashboard, f"   {cls_name}", (25, y),
                        FONT, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
            bar_filled = min(int((count / max(total_in, 1)) * 120), 120)
            cv2.rectangle(dashboard, (150, y - 10), (150 + 120, y - 2), DIVIDER_COLOR, -1)
            cv2.rectangle(dashboard, (150, y - 10), (150 + bar_filled, y - 2), INBOUND_COLOR, -1)
            cv2.putText(dashboard, str(count), (280, y),
                        FONT, 0.48, INBOUND_COLOR, 1, cv2.LINE_AA)

        y += 20
        cv2.line(dashboard, (15, y), (dash_w - 15, y), DIVIDER_COLOR, 1)
        y += 18

        # --- OUTBOUND section ---
        cv2.rectangle(dashboard, (15, y - 16), (140, y + 6), (50, 30, 130), -1)
        cv2.putText(dashboard, "  OUTBOUND", (15, y),
                    FONT, 0.5, OUTBOUND_COLOR, 1, cv2.LINE_AA)

        total_out_str = f"{total_out}"
        (tw, _), _ = cv2.getTextSize(total_out_str, FONT, 0.6, 2)
        cv2.putText(dashboard, total_out_str, (dash_w - 25 - tw, y),
                    FONT, 0.6, OUTBOUND_COLOR, 2, cv2.LINE_AA)

        y += 14
        for cls_name, count in counts["outbound"].items():
            y += 26
            cv2.putText(dashboard, f"   {cls_name}", (25, y),
                        FONT, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
            bar_filled = min(int((count / max(total_out, 1)) * 120), 120)
            cv2.rectangle(dashboard, (150, y - 10), (150 + 120, y - 2), DIVIDER_COLOR, -1)
            cv2.rectangle(dashboard, (150, y - 10), (150 + bar_filled, y - 2), OUTBOUND_COLOR, -1)
            cv2.putText(dashboard, str(count), (280, y),
                        FONT, 0.48, OUTBOUND_COLOR, 1, cv2.LINE_AA)

        # --- ปุ่ม RELOAD ---
        draw_rounded_rect(dashboard, (rld_x1, rld_y1), (rld_x2, rld_y2), BTN_RELOAD, radius=10)
        draw_text_centered(dashboard, "RELOAD STREAM", (rld_x1 + rld_x2) // 2, rld_y1 + 20,
                           FONT, 0.5, (255, 255, 255), 2)
        draw_text_centered(dashboard, "(or press  R)", (rld_x1 + rld_x2) // 2, rld_y1 + 36,
                           FONT, 0.35, (180, 220, 230), 1)

        # --- ปุ่ม STOP & EXPORT ---
        draw_rounded_rect(dashboard, (btn_x1, btn_y1), (btn_x2, btn_y2), BTN_STOP, radius=10)
        draw_text_centered(dashboard, "STOP & EXPORT", (btn_x1 + btn_x2) // 2, btn_y1 + 20,
                           FONT, 0.5, (255, 255, 255), 2)
        draw_text_centered(dashboard, "(or press  Q)", (btn_x1 + btn_x2) // 2, btn_y1 + 36,
                           FONT, 0.35, (180, 200, 255), 1)

        # --- Fast Mode indicator ---
        mode_label = "FAST MODE" if fast_mode else "QUALITY MODE"
        mode_color = (0, 200, 100) if fast_mode else (200, 140, 0)
        cv2.circle(dashboard, (22, dash_h - 12), 5, mode_color, -1)
        cv2.putText(dashboard, mode_label, (32, dash_h - 7),
                    FONT, 0.35, mode_color, 1, cv2.LINE_AA)

        cv2.imshow("Dashboard", dashboard)

        # Vehicle Counting frame — เพิ่ม overlay ข้อมูลมุมซ้ายบน
        overlay_h = 30
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (180, overlay_h), (10, 10, 20), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"IN:{total_in}  OUT:{total_out}  TOT:{total_all}",
                    (8, 20), FONT, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow("Vehicle Counting", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            stop_counting = True
            break
        elif key == ord('r'):
            manual_reload = True

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    if export_data and export_csv:
        try:
            with open(export_csv, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Vehicle_ID", "Class", "Direction"])
                writer.writerows(export_data)
            print(f"✅ บันทึกประวัติการนับลงไฟล์ CSV: {export_csv} แล้ว!")
            messagebox.showinfo("Success", f"นับเสร็จสิ้น! ข้อมูลบันทึกที่:\n{export_csv}")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการบันทึกไฟล์ CSV: {e}")
            messagebox.showerror("Error", f"บันทึกไฟล์ CSV ไม่ได้:\n{e}")


# ==========================================
# ส่วนที่ 3: Tkinter GUI — Dark Theme อัปเกรด
# ==========================================
def open_gui():
    # --- สี Palette ---
    BG         = "#12121e"
    PANEL      = "#1c1c2e"
    BORDER     = "#2e2e4a"
    ACCENT     = "#00c8ff"
    BTN_GREEN  = "#1db954"
    BTN_HOVER  = "#17a046"
    TEXT       = "#e0e0ee"
    MUTED      = "#7070a0"
    ENTRY_BG   = "#1a1a2e"
    ENTRY_FG   = "#e0e0ee"

    root = tk.Tk()
    root.title("Vehicle Counter  —  YOLOv8")
    root.geometry("600x400")
    root.resizable(False, False)
    root.configure(bg=BG)

    # ============== FONTS ================
    FONT_TITLE  = ("Segoe UI", 18, "bold")
    FONT_HEAD   = ("Segoe UI", 10, "bold")
    FONT_BODY   = ("Segoe UI", 9)
    FONT_SMALL  = ("Segoe UI", 8)
    FONT_BTN    = ("Segoe UI", 11, "bold")

    # ============== STYLE ================
    style = ttk.Style(root)
    style.theme_use("clam")

    # ตัวแปรเก็บค่าตัวเลือก
    source_type   = tk.IntVar(value=1)
    file_path_var = tk.StringVar()
    stream_url_var= tk.StringVar(value="")
    csv_path_var  = tk.StringVar(value="vehicle_counts.csv")
    fast_mode_var = tk.BooleanVar(value=True)

    def browse_file():
        path = filedialog.askopenfilename(
            title="เลือกไฟล์วิดีโอ",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if path:
            file_path_var.set(path)

    def start_processing():
        export_csv = csv_path_var.get().strip() or "vehicle_counts.csv"

        if source_type.get() == 1:
            source = file_path_var.get().strip()
            if not source:
                messagebox.showwarning("Warning", "กรุณาเลือกไฟล์วิดีโอ")
                return
        else:
            raw = stream_url_var.get().strip()
            source = raw.split('\n')[0].split('\r')[0]
            if not source:
                messagebox.showwarning("Warning", "กรุณากรอก Streaming URL หรือ Camera ID (เช่น 0)")
                return

        fast_mode = fast_mode_var.get()
        root.destroy()
        run_counter(source, export_csv, fast_mode)

    def on_enter_btn(e):
        start_btn.configure(bg=BTN_HOVER)

    def on_leave_btn(e):
        start_btn.configure(bg=BTN_GREEN)

    # ======== HEADER =========
    header = tk.Frame(root, bg=PANEL, height=64)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(header, text="🚗  VEHICLE COUNTER", font=FONT_TITLE,
             bg=PANEL, fg=ACCENT).pack(side="left", padx=22, pady=14)
    tk.Label(header, text="YOLOv8  •  Real-time", font=FONT_SMALL,
             bg=PANEL, fg=MUTED).pack(side="right", padx=22, pady=22)

    # เส้นคั่น accent
    tk.Frame(root, bg=ACCENT, height=2).pack(fill="x")

    # ======== MAIN CONTENT =========
    content = tk.Frame(root, bg=BG)
    content.pack(fill="both", expand=True, padx=24, pady=12)

    # --- Source Selection ---
    tk.Label(content, text="VIDEO SOURCE", font=FONT_HEAD,
             bg=BG, fg=ACCENT).grid(row=0, column=0, sticky="w", pady=(4, 6))

    # Radio — Video File
    tk.Radiobutton(content, text="อัปโหลดไฟล์วิดีโอ  (Video File)",
                   variable=source_type, value=1,
                   bg=BG, fg=TEXT, selectcolor=PANEL,
                   activebackground=BG, activeforeground=ACCENT,
                   font=FONT_BODY).grid(row=1, column=0, sticky="w")

    file_frame = tk.Frame(content, bg=BG)
    file_frame.grid(row=2, column=0, sticky="ew", pady=(2, 8), padx=(16, 0))

    file_entry = tk.Entry(file_frame, textvariable=file_path_var,
                          bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ACCENT,
                          relief="flat", font=FONT_BODY, width=40,
                          highlightthickness=1, highlightbackground=BORDER,
                          highlightcolor=ACCENT)
    file_entry.pack(side="left", ipady=5, padx=(0, 6))

    browse_btn = tk.Button(file_frame, text="Browse…", command=browse_file,
                           bg=PANEL, fg=ACCENT, activebackground=BORDER,
                           activeforeground=ACCENT, relief="flat", font=FONT_BODY,
                           cursor="hand2", padx=10, pady=4)
    browse_btn.pack(side="left")

    # Radio — Stream / Camera
    tk.Radiobutton(content, text="ลิงก์สตรีม / กล้องวงจรปิด  (Streaming URL / Camera ID)",
                   variable=source_type, value=2,
                   bg=BG, fg=TEXT, selectcolor=PANEL,
                   activebackground=BG, activeforeground=ACCENT,
                   font=FONT_BODY).grid(row=3, column=0, sticky="w", pady=(4, 0))

    stream_frame = tk.Frame(content, bg=BG)
    stream_frame.grid(row=4, column=0, sticky="ew", pady=(2, 12), padx=(16, 0))

    stream_entry = tk.Entry(stream_frame, textvariable=stream_url_var,
                            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ACCENT,
                            relief="flat", font=FONT_BODY, width=50,
                            highlightthickness=1, highlightbackground=BORDER,
                            highlightcolor=ACCENT)
    stream_entry.pack(side="left", ipady=5)

    # เส้นคั่น
    tk.Frame(content, bg=BORDER, height=1).grid(row=5, column=0, sticky="ew", pady=4)

    # --- Settings Row ---
    settings_frame = tk.Frame(content, bg=BG)
    settings_frame.grid(row=6, column=0, sticky="ew", pady=6)

    # Fast Mode Toggle
    fast_chk = tk.Checkbutton(settings_frame,
                               text="⚡  Fast Mode  (ความเร็วสูงสุด — ลดติ๊กเพื่อเพิ่มความชัด)",
                               variable=fast_mode_var,
                               bg=BG, fg=TEXT, selectcolor=PANEL,
                               activebackground=BG, activeforeground=ACCENT,
                               font=FONT_BODY)
    fast_chk.pack(side="left")

    # CSV Row
    csv_row = tk.Frame(content, bg=BG)
    csv_row.grid(row=7, column=0, sticky="ew", pady=(6, 0))

    tk.Label(csv_row, text="📄  Export CSV:", font=FONT_BODY,
             bg=BG, fg=MUTED).pack(side="left", padx=(0, 8))

    csv_entry = tk.Entry(csv_row, textvariable=csv_path_var,
                         bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ACCENT,
                         relief="flat", font=FONT_BODY, width=28,
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT)
    csv_entry.pack(side="left", ipady=4)

    # ======== START BUTTON =========
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(fill="x", padx=24, pady=(4, 18))

    start_btn = tk.Button(btn_frame, text="🚀  เริ่มนับยานพาหนะ  (Start)",
                          font=FONT_BTN, bg=BTN_GREEN, fg="white",
                          activebackground=BTN_HOVER, activeforeground="white",
                          relief="flat", cursor="hand2", padx=20, pady=10,
                          command=start_processing)
    start_btn.pack(fill="x")
    start_btn.bind("<Enter>", on_enter_btn)
    start_btn.bind("<Leave>", on_leave_btn)

    root.mainloop()


if __name__ == "__main__":
    open_gui()
