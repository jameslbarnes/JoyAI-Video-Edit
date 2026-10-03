"""Real loopback wire/cadence tests with CPU frames; no live stream/GPU work."""
import json
import socket
import struct
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from xvideo.serving.direct_ndi_output import DirectNdiPublisher
from xvideo.serving.upscale_output import OutputUpscale


class DirectOutput(unittest.TestCase):
    def test_second_scene_preparation_cannot_finish_previous_handoff(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                p=DirectNdiPublisher()
                frame=np.zeros((360,648,3),np.uint8)
                p.last_frame,p.last_key=frame,(648,360,24.)
                p.begin_transition(lambda:frame,moving=moving)
                p.reset_scene()
                p.frames.extend((frame,p.last_key) for _ in range(32))
                # A second request arrives before the first handoff finishes.
                p.begin_transition(lambda:frame,moving=moving)
                start=p.transition['start']
                p._next_frame(start+1)
                p._next_frame(start+1.3)
                self.assertIsNotNone(p.transition)
                self.assertFalse(p.transition['reset'])
                self.assertTrue(p.reset_scene())
                self.assertEqual(p.transitions_completed,0)
                self.assertEqual(len(p.frames),0)

    def test_initial_reserve_and_underrun_accounting(self):
        p=DirectNdiPublisher()
        frame=np.zeros((360,648,3),np.uint8);key=(648,360,24.)
        p.frames.extend((frame,key) for _ in range(9))
        self.assertIsNone(p._next_frame(0));self.assertEqual(len(p.frames),9)
        p.frames.extend((frame,key) for _ in range(8))
        self.assertIsNotNone(p._next_frame(1));self.assertTrue(p.playback_primed)
        p.frames.clear();p._record_underrun();p._record_underrun()
        self.assertEqual(p.underruns,1)
        p.frames.append((frame,key));p._next_frame(2);p._record_underrun()
        self.assertEqual(p.underruns,2)
    def test_moving_bridge_blends_advancing_shader_and_generated_frames(self):
        p = DirectNdiPublisher()
        old = np.zeros((360,648,3),np.uint8)
        source = np.full_like(old,100)
        p.last_frame,p.last_key=old,(648,360,32.)
        p.begin_transition(lambda:source,moving=True)
        start=p.transition['start'];p.reset_scene()
        p.frames.extend((np.full_like(old,i),p.last_key) for i in range(20,52))
        self.assertEqual(int(p._next_frame(start+1)[0][0,0,0]),100)
        source=np.full_like(old,200)
        self.assertEqual(int(p._next_frame(start+1.1)[0][0,0,0]),110)
        self.assertEqual(int(p._next_frame(start+1.7)[0][0,0,0]),22)
        self.assertIsNone(p.transition)
        self.assertEqual(p.transitions_completed,1)
        self.assertEqual(len(p.frames),29)

    def test_first_scene_frame_stays_hidden_until_motion_reserve_is_ready(self):
        p = DirectNdiPublisher()
        old = np.zeros((360,648,3),np.uint8)
        shader = np.full_like(old,40)
        generated = np.full_like(old,240)
        p.last_frame,p.last_key=old,(648,360,24.)
        p.begin_transition(lambda:shader,moving=True)
        start=p.transition['start'];p.reset_scene()
        p.frames.append((generated,p.last_key))
        self.assertEqual(int(p._next_frame(start+1)[0][0,0,0]),40)
        shader=np.full_like(old,80)
        self.assertEqual(int(p._next_frame(start+1.3)[0][0,0,0]),80)
        self.assertIsNone(p.transition['in_start'])
        self.assertEqual(len(p.frames),1)
        p.frames.extend((generated,p.last_key) for _ in range(8))
        self.assertEqual(int(p._next_frame(start+1.4)[0][0,0,0]),80)
        self.assertIsNone(p.transition['in_start'])
        self.assertEqual(len(p.frames),9)
        p.frames.extend((np.full_like(old,i),p.last_key) for i in range(90,98))
        self.assertEqual(int(p._next_frame(start+1.5)[0][0,0,0]),80)
        self.assertIsNotNone(p.transition['in_start'])
        self.assertEqual(len(p.frames),16)
        self.assertEqual(int(p._next_frame(start+1.8)[0][0,0,0]),240)
        self.assertIsNone(p.transition)

    def test_transition_fades_to_live_shader_and_waits_for_real_following_chunk(self):
        p = DirectNdiPublisher()
        old = np.zeros((360,648,3), np.uint8)
        shader = np.full_like(old, 200)
        new = np.full_like(old, 50)
        p.last_frame, p.last_key = old, (648,360,32.)
        self.assertTrue(p.begin_transition(lambda: shader))
        start = p.transition['start']
        self.assertEqual(int(p._next_frame(start)[0][0,0,0]), 0)
        self.assertEqual(int(p._next_frame(start+.3)[0][0,0,0]), 100)
        self.assertTrue(p.reset_scene())
        self.assertEqual(int(p._next_frame(start+.7)[0][0,0,0]), 200)
        shader = np.full_like(old, 180)
        self.assertEqual(int(p._next_frame(start+.8)[0][0,0,0]), 180)
        # The synthetic morph alone cannot end the bridge and run dry.
        p.frames.extend((new, p.last_key) for _ in range(16))
        self.assertEqual(int(p._next_frame(start+.9)[0][0,0,0]), 180)
        self.assertEqual(len(p.frames), 16)
        p.frames.extend((new, p.last_key) for _ in range(16))
        self.assertEqual(int(p._next_frame(start+1)[0][0,0,0]), 50)
        self.assertIsNone(p.transition)
        self.assertEqual(p.transitions_completed, 1)

    def test_transition_cancel_expiry_and_close_release_source(self):
        p = DirectNdiPublisher()
        self.assertFalse(p.begin_transition(lambda: None))
        p.last_frame = np.zeros((360,648,3), np.uint8)
        p.last_key = (648,360,32.)
        p.begin_transition(lambda: None)
        self.assertIsNone(p._next_frame(p.transition['start']+21))
        self.assertIsNone(p.transition)
        p.begin_transition(lambda: None)
        p.cancel_transition()
        self.assertFalse(p.reset_scene())
        p.begin_transition(lambda: None)
        p.close()
        self.assertIsNone(p.transition)
        self.assertIsNone(p.last_frame)

    def test_morph_handoff_has_headroom_for_next_chunk(self):
        p = DirectNdiPublisher(prime_s=10)
        frame = np.zeros((360,648,3), np.uint8)
        p.last_frame, p.last_key = frame, (648,360,32.)
        try:
            p.begin_transition(lambda: frame)
            start = p.transition['start']
            p.reset_scene()
            p.submit([frame]*16, 32)  # Morph plus its endpoint.
            p.submit([frame]*16, 32)  # Following real chunk makes it playable.
            with p.condition:
                p._next_frame(start+1)
            p.submit([frame]*16, 32)  # A fast following chunk cannot clip the morph.
            self.assertEqual(p.status()['queued'], 47)
            self.assertEqual(p.status()['dropped'], 0)
            self.assertLessEqual(p.status()['queued'], p.max_frames)
        finally:
            p.close()

    def test_wire_order_pacing_and_shutdown(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(3)
        received, errors = [], []

        def receive():
            try:
                with listener.accept()[0] as sock:
                    sock.settimeout(3)
                    with sock.makefile('rb') as stream:
                        hello = json.loads(stream.readline())
                        self.assertEqual(hello['frameRateN'], 16000)
                        self.assertEqual(hello['senderId'], 'primary-joyai')
                        for _ in range(8):
                            header = stream.read(20)
                            magic, version, fmt, _, w, h, size = struct.unpack('<4sBBHIII', header)
                            self.assertEqual((magic, version, fmt, w, h, size), (b'EVSF', 1, 1, 648, 360, 648*360*4))
                            pixels = stream.read(size)
                            self.assertEqual(pixels[3], 255)
                            received.append((pixels[0], time.monotonic()))
                        self.assertEqual(stream.read(1), b'')
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=receive)
        worker.start()
        publisher = DirectNdiPublisher(port=listener.getsockname()[1], prime_s=.03)
        try:
            publisher.submit([np.full((360,648,3), i, np.uint8) for i in range(8)], 16)
            deadline = time.monotonic() + 3
            while publisher.status()['sent'] < 8 and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(publisher.status()['sent'], 8)
        finally:
            publisher.close()
            worker.join(3)
            listener.close()
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual([x[0] for x in received], list(range(8)))
        self.assertGreater(received[-1][1] - received[0][1], .35)
        self.assertLess(received[-1][1] - received[0][1], 1)
        self.assertIsNone(publisher.thread)

    def test_transport_error_recovers_after_retry_delay(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(2)
        listener.settimeout(3)
        received = []

        def serve():
            first, _ = listener.accept()
            first.close()  # bridge drops the first connection mid-stream
            with listener.accept()[0] as sock:
                sock.settimeout(3)
                buf = sock.makefile('rb')
                hello = json.loads(buf.readline())
                header = buf.read(20)
                received.append((hello['width'], struct.unpack('<4sBBHIII', header)[0]))

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        publisher = DirectNdiPublisher(port=listener.getsockname()[1], prime_s=0, retry_s=.1)
        frame = np.zeros((360, 648, 3), np.uint8)
        try:
            deadline = time.monotonic() + 5
            while publisher.error is None and time.monotonic() < deadline:
                if publisher.available():
                    publisher.submit([frame] * 4, 16)
                time.sleep(.02)
            self.assertIsNotNone(publisher.error)
            self.assertFalse(publisher.available())  # retry delay not yet elapsed
            while not received and time.monotonic() < deadline:
                if publisher.available():
                    publisher.submit([frame] * 4, 16)
                time.sleep(.05)
            self.assertEqual(received, [(648, b'EVSF')])
            status = publisher.status()
            self.assertGreaterEqual(status['reconnects'], 1)
            self.assertIsNotNone(status['last_error'])
        finally:
            publisher.close()
            listener.close()
            server.join(2)

    def test_shader_bridge_keeps_same_socket_and_cadence_during_reset(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(3)
        received, errors = [], []
        def receive():
            try:
                with listener.accept()[0] as sock, sock.makefile('rb') as stream:
                    sock.settimeout(3)
                    json.loads(stream.readline())
                    while True:
                        header = stream.read(20)
                        if not header:
                            break
                        _, _, _, _, w, h, size = struct.unpack('<4sBBHIII', header)
                        self.assertEqual((w,h), (648,360))
                        pixels = stream.read(size)
                        received.append((pixels[0], time.perf_counter()))
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=receive)
        worker.start()
        p = DirectNdiPublisher(port=listener.getsockname()[1], prime_s=0)
        frame = np.zeros((360,648,3), np.uint8)
        try:
            p.submit([frame] * 16, 32)
            deadline = time.perf_counter()+2
            while p.status()['sent'] < 3 and time.perf_counter() < deadline:
                time.sleep(.005)
            original_socket = p.socket
            self.assertTrue(p.begin_transition(lambda: np.full_like(frame, 180)))
            self.assertTrue(p.reset_scene())
            time.sleep(.8)
            self.assertIs(p.socket, original_socket)
            self.assertGreater(p.status()['transition_frames'], 20)
            p.submit([np.full_like(frame,50)]*32, 32)
            time.sleep(.15)
            self.assertEqual(p.status()['transitions_completed'], 1)
        finally:
            p.close()
            worker.join(3)
            listener.close()
        self.assertFalse(errors, errors)
        self.assertTrue(any(value == 180 for value, _ in received))
        self.assertEqual(received[-1][0], 50)
        self.assertLess(max(b[1]-a[1] for a,b in zip(received,received[1:])), .15)

    def test_queue_bounded_and_shape_change_discards_old_frames(self):
        p = DirectNdiPublisher(max_frames=16, prime_s=10)
        frame = np.zeros((360,648,3), np.uint8)
        try:
            for _ in range(3):
                p.submit([frame] * 16, 32)
            self.assertEqual(p.status()['queued'], 16)
            self.assertEqual(p.status()['dropped'], 32)
            p.submit([np.zeros((1440,2592,3), np.uint8)], 32)
            self.assertEqual(p.status()['queued'], 1)
            self.assertEqual(p.status()['dropped'], 48)
        finally:
            p.close()

    def test_connection_failure_latches_and_stops_accepting(self):
        p = DirectNdiPublisher(prime_s=0)
        frame = np.zeros((360,648,3), np.uint8)
        with patch('xvideo.serving.direct_ndi_output.socket.create_connection', side_effect=OSError('test unavailable')):
            try:
                p.submit([frame]*16, 32)
                p.thread.join(2)
                self.assertIn('test unavailable', p.status()['error'])
                self.assertFalse(p.status()['enabled'])
                self.assertFalse(p.submit([frame], 32))
                self.assertEqual(p.status()['queued'], 0)
            finally:
                p.close()

    def test_full_resolution_publication_and_smaller_preview(self):
        class Publisher:
            error = None
            thread = None
            def submit(self, frames, fps):
                self.shape, self.fps = frames[0].shape, fps
            def available(self):
                return not self.error
            def status(self):
                return {'error': self.error}
        output = OutputUpscale.__new__(OutputUpscale)
        output.direct_requested = True
        output.direct_publisher = Publisher()
        profile = {'presentation_fps': 32}
        with ThreadPoolExecutor(max_workers=1) as output.encoder:
            for shape in ((1080,1944,3), (1440,2592,3)):
                jpeg = output.encode_frames([np.zeros(shape, np.uint8)], 95, profile)[0]
                self.assertEqual(output.direct_publisher.shape, shape)
                self.assertEqual(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape, (720,1296,3))
                self.assertTrue(profile['direct_ndi']['enabled'])
            output.direct_publisher.error = 'failed'
            output.encode_frames([np.zeros((1080,1944,3), np.uint8)], 95, profile)
            self.assertFalse(profile['direct_ndi']['enabled'])


if __name__ == '__main__':
    unittest.main()
