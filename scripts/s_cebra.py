#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from collections import deque


# camara frontal inferior = 3


class ZebraStripeCountNode(Node):
    def __init__(self):
        super().__init__('Cebra_node')
        self.bridge = CvBridge()

        self.lock = threading.Lock()
        self.frame = None

        self.camera_topic = '/camera/csi_image_3'
        self.create_subscription(Image, self.camera_topic, self.cb_img, 10)

        # no parpadeo
        self.votes = deque(maxlen=7)
        self.last_print = None

        self.debug = True

        # ROI centro-inferior 
        self.roi_top = 0.80
        self.roi_bottom = 0.985
        self.roi_width = 0.40

        # franjas
        self.min_stripes = 4
        self.max_stripes = 7

        t = threading.Thread(target=self.loop, daemon=True)
        t.start()

        self.get_logger().info(f'Cebra iniciado - topic: {self.camera_topic}')

    def cb_img(self, msg: Image):
        if not msg.data:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if img is None or img.size == 0:
                return
            with self.lock:
                self.frame = img
        except Exception:
            return

    def loop(self):
        while rclpy.ok():
            with self.lock:
                img = None if self.frame is None else self.frame.copy()

            if img is not None:
                present_raw, stripes_est = self.detect(img)

                self.votes.append(1 if present_raw else 0)
                present = (sum(self.votes) >= 5)

                # stop / go
                out = present
                if out != self.last_print:
                    self.last_print = out
                    action = "stop" if present else "go"
                    self.get_logger().info(f'crosswalk present={present} | {action}')

            threading.Event().wait(0.05)

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]

        rt = int(h * self.roi_top)
        rb = int(h * self.roi_bottom)

        roi_w = int(w * self.roi_width)
        cx = w // 2
        rl = max(0, cx - roi_w // 2)
        rr = min(w, cx + roi_w // 2)

        roi = img_bgr[rt:rb, rl:rr]
        if roi.size == 0:
            return False, -1

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # banda inferior, ROI
        band_top = int(gray.shape[0] * 0.20)
        band = gray[band_top:, :]

        band_blur = cv2.GaussianBlur(band, (5, 5), 0)

        # binariza blanco/negro
        _, bw = cv2.threshold(band_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bw = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1
        )

        H = bw.shape[0]
        sample_rows = np.linspace(int(H * 0.65), int(H * 0.95), 8).astype(int)

        trans_counts = []
        for y in sample_rows:
            row = (bw[y, :] > 0).astype(np.uint8)
            trans_counts.append(int(np.sum(row[1:] != row[:-1])))

        trans_counts = np.array(trans_counts, dtype=np.int32)
        trans_med = float(np.median(trans_counts))

        # Estimación de franjas 
        stripes_est = int(np.floor(trans_med + 1))

        # desicion
        present = (self.min_stripes <= stripes_est <= self.max_stripes)

        # print “# stripes”
        if self.debug:
            dbg = img_bgr.copy()
            cv2.rectangle(dbg, (rl, rt), (rr, rb), (0, 255, 255), 2)
            cv2.putText(
                dbg,
                f"stripes~{stripes_est}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )
            cv2.imshow("dbg_roi", dbg)
            cv2.imshow("dbg_bw_band", bw)
            cv2.waitKey(1)

        return present, stripes_est


def main(args=None):
    rclpy.init(args=args)
    node = ZebraStripeCountNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.debug:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

