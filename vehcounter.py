import cv2
import argparse
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
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
        # รับค่า param ที่เป็นตัวแปร (btn_x1, btn_y1, btn_x2, btn_y2, rld_x1, rld_y1, rld_x2, rld_y2) 
        btn_x1, btn_y1, btn_x2, btn_y2, rld_x1, rld_y1, rld_x2, rld_y2 = param
        
        # เช็คปุ่ม Stop
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            stop_counting = True
            
        # เช็คปุ่ม Reload
        if rld_x1 <= x <= rld_x2 and rld_y1 <= y <= rld_y2:
            manual_reload = True

def run_counter(source, export_csv, fast_mode):
    """
    ฟังก์ชันหลักในการนับยานพาหนะเมื่อได้ Source และ Export Path แล้ว
    """
    global line_pts, stop_counting
    
    print(f"กำลังเปิดโหลดโมเดลและ Source: {source}")
    # โหลดโมเดล YOLOv8
    model = YOLO("yolov8n.pt") 

    # คลาส COCO สำหรับยานพาหนะ
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
        # ลองตั้งค่า buffer size เล็กๆ เผื่อช่วยเรื่อง realtime
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # เปิดวิดีโอครั้งแรก
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

    # คำนวณ Scale สำหรับย่อภาพ
    # ปรับ max_height ลดลงมาเหลือ 480 เพื่อให้กวาดภาพบน CPU ได้เร็วขึ้นมาก
    h, w = frame.shape[:2]
    max_height = 480
    scale = 1.0
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = w, h
        
    # Resize เฟรมแรกสำหรับตั้งค่าเส้นให้เล็กลงด้วย
    frame = cv2.resize(frame, (new_w, new_h))
        
    # ==========================================
    # ส่วนที่ 1: UI สำหรับคลิกวาดเส้นบน OpenCV (Setup Line)
    # ==========================================
    cv2.namedWindow("Setup Line")
    cv2.setMouseCallback("Setup Line", draw_line)
    
    print("===== กำลังตั้งค่าเส้นสมมติ =====")
    
    while True:
        temp_frame = frame.copy()
        
        # วาดเส้นถ้ามีการคลิกจุด
        if len(line_pts) >= 1:
            cv2.circle(temp_frame, line_pts[0], 5, (0, 255, 255), -1)
            
        # ลากเส้นนำสายตาระหว่างคลิกจุดที่ 1 ไปยังตำแหน่งที่เมาส์อยู่ปัจจุบัน
        if len(line_pts) == 1 and current_mouse_pos is not None:
            cv2.line(temp_frame, line_pts[0], current_mouse_pos, (0, 255, 255), 1)
            
        if len(line_pts) == 2:
            cv2.line(temp_frame, line_pts[0], line_pts[1], (0, 255, 255), 2)
            cv2.putText(temp_frame, "Counting Line", (line_pts[0][0], line_pts[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        cv2.rectangle(temp_frame, (5, 5), (600, 40), (0, 0, 0), -1)
        cv2.putText(temp_frame, "Click 2 points to draw a line. Press 'c' to confirm, 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
    # ส่วนที่ 2: เริ่มตรวจจับยานพาหนะและนับจำนวน
    # ==========================================
    cv2.namedWindow("Vehicle Counting")
    cv2.namedWindow("Dashboard")
    
    # สร้างแคนวาสสำหรับ Dashboard (กว้าง 400, สูง 500)
    dash_w, dash_h = 400, 500
    
    # พิกัดของปุ่มหยุด ในหน้า Dashboard
    btn_w, btn_h = 160, 40
    btn_x1, btn_y1 = (dash_w // 2) - (btn_w // 2), dash_h - btn_h - 20
    btn_x2, btn_y2 = btn_x1 + btn_w, btn_y1 + btn_h
    
    # พิกัดของปุ่ม Reload (วางไว้บนปุ่ม Stop)
    rld_w, rld_h = 160, 40
    rld_x1, rld_y1 = btn_x1, btn_y1 - rld_h - 15
    rld_x2, rld_y2 = btn_x1 + rld_w, rld_y1 + rld_h
    
    # กำหนด Mouse Callback ให้ส่งค่าพิกัดปุ่มไปด้วย บนหน้าหน้า Dashboard
    cv2.setMouseCallback("Dashboard", click_stop_button, param=(btn_x1, btn_y1, btn_x2, btn_y2, rld_x1, rld_y1, rld_x2, rld_y2))

    global manual_reload
    last_frame_time = datetime.now()

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
        
        # ตรวจสอบว่าเฟรมค้าง (ไม่มานานเกิน 5 วินาที) หรือกดปุ่ม Reload ดึง stream ใหม่
        time_since_last_frame = (datetime.now() - last_frame_time).total_seconds()
        
        is_stalled = (not ret and not str(source).isdigit()) or (time_since_last_frame > 5.0 and not str(source).isdigit())
        
        if is_stalled or manual_reload:
            if manual_reload:
                print("ผู้ใช้กดปุ่ม Reload! กำลังโหลด Stream ใหม่...")
                manual_reload = False
            else:
                print("สตรีมมิ่งค้างหรือไม่ตอบสนอง... กำลังตั้งค่าเชื่อมต่อสตรีมใหม่ (Auto-Reload)")
                
            cap = connect_stream()
            last_frame_time = datetime.now() # Reset เวลา
            continue
            
        if not ret:
            # ถ้าเป็นไฟล์วิดีโอปกติ และอ่านไม่ขึ้นแล้ว แสดงว่าจบไฟล์
            if str(source).isdigit() or source.endswith('.mp4') or source.endswith('.avi'):
                print("สิ้นสุดไฟล์วิดีโอ")
                break
            continue

        # รีเซ็ตเวลาหลังจากอ่านเฟรมสำเร็จ
        last_frame_time = datetime.now()

        # ย่อขนาดเฟรมระหว่างประมวลผลให้เท่ากับตอนตั้งค่า
        frame = cv2.resize(frame, (new_w, new_h))

        # เลือกว่าจะใช้ความละเอียดเท่าไร (imgsz)
        # 320 เน้นประมวลผลเร็วสุดๆ (FPS สูง)
        # 640 เน้นความแม่นยำ (FPS ตก)
        tracking_imgsz = 320 if fast_mode else 640
        
        results = model.track(frame, persist=True, classes=vehicle_classes, imgsz=tracking_imgsz, verbose=False)

        cv2.line(frame, line_start, line_end, (0, 255, 255), 2)
        cv2.putText(frame, "Counting Line", (line_start[0], line_start[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
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
                label = f"{cls_name} ID:{track_id}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                for i in range(1, len(track_history[track_id])):
                    cv2.line(frame, track_history[track_id][i-1], track_history[track_id][i], (255, 0, 0), 2)

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

        # สร้างภาพพื้นหลังสีดำสำหรับหน้าต่าง Dashboard
        dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
        
        # วาดข้อความสรุปผล(Text) บน Dashboard
        y_offset = 40
        cv2.putText(dashboard, "--- INBOUND ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        for cls_name, count in counts["inbound"].items():
            y_offset += 30
            cv2.putText(dashboard, f"{cls_name}: {count}", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y_offset += 40
        cv2.putText(dashboard, "--- OUTBOUND ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        for cls_name, count in counts["outbound"].items():
            y_offset += 30
            cv2.putText(dashboard, f"{cls_name}: {count}", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # วาดปุ่ม "STOP & EXPORT" สีแดง
        cv2.rectangle(dashboard, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 255), -1)
        cv2.putText(dashboard, "STOP & EXPORT", (btn_x1 + 10, btn_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(dashboard, "(or 'q')", (btn_x1 + 55, btn_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # วาดปุ่ม "RELOAD" สีส้ม
        cv2.rectangle(dashboard, (rld_x1, rld_y1), (rld_x2, rld_y2), (0, 165, 255), -1)
        cv2.putText(dashboard, "RELOAD STREAM", (rld_x1 + 10, rld_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(dashboard, "(or 'r')", (rld_x1 + 55, rld_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Dashboard", dashboard)
        cv2.imshow("Vehicle Counting", frame)
        
        # สำหรับสตรีมมิ่ง กด 'q' หรือ 'ESC' เพื่อบังคับหยุดเช่นกัน และ 'r' เพื่อ reload
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
# ส่วนที่ 3: ระบบ GUI หน้าหลัก (Tkinter) สำหรับเลือก Source
# ==========================================
def open_gui():
    root = tk.Tk()
    root.title("Vehicle Counter Setup")
    root.geometry("550x300")
    
    # ตัวแปรเก็บค่าตัวเลือกการโหลด
    source_type = tk.IntVar(value=1) # 1=File, 2=Stream/Webcam
    file_path_var = tk.StringVar()
    stream_url_var = tk.StringVar(value="") # ค่า Default สำหรับ Stream ลบ "0" ออก
    csv_path_var = tk.StringVar(value="vehicle_counts.csv")
    fast_mode_var = tk.BooleanVar(value=True) # ตัวแปรโหมดความเร็ว (ค่าเริ่มต้นคือเร็ว)

    def browse_file():
        file_path = filedialog.askopenfilename(
            title="เลือกไฟล์วิดีโอ",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            file_path_var.set(file_path)

    def start_processing():
        export_csv = csv_path_var.get().strip()
        if not export_csv:
            export_csv = "vehicle_counts.csv"
            
        if source_type.get() == 1:
            source = file_path_var.get().strip()
            if not source:
                messagebox.showwarning("Warning", "กรุณาเลือกไฟล์วิดีโอ")
                return
        else:
            raw_source = stream_url_var.get().strip()
            # ตัดเอาเฉพาะบรรทัดแรกในกรณีที่มีคนเผลอวางข้อความยาวๆ ที่มีเว้นบรรทัด
            source = raw_source.split('\n')[0].split('\r')[0]
            if not source:
                messagebox.showwarning("Warning", "กรุณากรอก Streaming URL หรือ ID กล้อง (เช่น 0)")
                return

        # ปิดหน้าต่าง GUI (ลบ root) แล้วไปรัน OpenCV ต่อ
        fast_mode = fast_mode_var.get()
        root.destroy()
        run_counter(source, export_csv, fast_mode)

    # วาดหน้าต่าง UI
    tk.Label(root, text="YOLOv8 Vehicle Counter", font=("Helvetica", 16, "bold")).pack(pady=10)

    # Frame สำหรับ Radio Button
    radio_frame = tk.Frame(root)
    radio_frame.pack(fill="x", padx=20)

    tk.Radiobutton(radio_frame, text="อัปโหลดไฟล์วิดีโอ (Video File)", variable=source_type, value=1).pack(anchor="w")
    file_frame = tk.Frame(radio_frame)
    file_frame.pack(fill="x", padx=20, pady=2)
    tk.Entry(file_frame, textvariable=file_path_var, width=40).pack(side="left")
    tk.Button(file_frame, text="Browse", command=browse_file).pack(side="left", padx=5)

    tk.Radiobutton(radio_frame, text="ลิงก์สตรีม / กล้องวงจรปิด (Streaming URL / Camera ID)", variable=source_type, value=2).pack(anchor="w", pady=(10,0))
    stream_frame = tk.Frame(radio_frame)
    stream_frame.pack(fill="x", padx=20, pady=2)
    tk.Entry(stream_frame, textvariable=stream_url_var, width=50).pack(side="left")

    # ตัวเลือกปรับระดับความเร็วและคุณภาพ
    quality_frame = tk.Frame(root)
    quality_frame.pack(fill="x", padx=20, pady=5)
    tk.Checkbutton(quality_frame, text="✅ ใช้งานโหมดความเร็วสูงสุด (Fast Mode - เลิกติ๊กเพื่อให้ภาพชัดขึ้น)", 
                   variable=fast_mode_var).pack(anchor="w")

    # ตั้งชื่อไฟล์ CSV
    csv_frame = tk.Frame(root)
    csv_frame.pack(fill="x", padx=20, pady=10)
    tk.Label(csv_frame, text="ตั้งชื่อไฟล์ CSV (Export):").pack(side="left")
    tk.Entry(csv_frame, textvariable=csv_path_var, width=30).pack(side="left", padx=5)

    # ปุ่มเริ่มทำงาน
    tk.Button(root, text="🚀 เริ่มนับยานพาหนะ (Start)", font=("Helvetica", 12), bg="green", fg="white", command=start_processing).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    # เปิดหน้าต่าง GUI ตัวเริ่มต้น
    open_gui()
