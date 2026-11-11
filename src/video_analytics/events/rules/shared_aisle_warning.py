"""Regra experimental de circulação partilhada pessoa–veículo.

Este módulo trabalha em coordenadas da imagem. Não estima distância física,
velocidade real, risco de colisão ou near-miss.
"""

from collections import deque
from dataclasses import dataclass
from itertools import product
from math import hypot

import cv2
import numpy as np

from video_analytics.tracking.models import TrackedObject


@dataclass(frozen=True)
class SharedAisleWarningResult:
    """Evidência observada quando uma interação satisfaz a regra."""

    frame_index: int
    source_time_seconds: float | None
    person_track_id: int
    vehicle_track_id: int
    confidence: float
    person_bottom_center: tuple[int, int]
    vehicle_bottom_center: tuple[int, int]
    consecutive_frames: int
    vehicle_image_displacement_px: float


class SharedAisleWarningRule:
    """Detetar circulação partilhada com movimento aparente do veículo.

    Uma nova notificação só pode ocorrer quando a interação anterior deixa
    de estar presente durante ``rearm_after_frames`` consecutivos. Isto evita
    repetição imediata após breves falhas de tracking; não corrige os IDs.
    """

    def __init__(
        self,
        polygon_pixels: list[list[int]],
        confirmation_frames: int = 3,
        motion_window_frames: int = 4,
        min_vehicle_displacement_px: float = 12.0,
        rearm_after_frames: int = 20,
    ) -> None:
        polygon = np.asarray(polygon_pixels, dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("polygon_pixels requer pelo menos 3 pontos [x, y].")
        if confirmation_frames < 1 or motion_window_frames < 2:
            raise ValueError("confirmation_frames >= 1 e motion_window_frames >= 2.")
        if min_vehicle_displacement_px < 0 or rearm_after_frames < 1:
            raise ValueError("Deslocamento >= 0 e rearm_after_frames >= 1.")

        self.polygon_pixels = polygon
        self.confirmation_frames = confirmation_frames
        self.motion_window_frames = motion_window_frames
        self.min_vehicle_displacement_px = min_vehicle_displacement_px
        self.rearm_after_frames = rearm_after_frames

        self.frame_index = -1
        self.pair_sequences: dict[tuple[int, int], int] = {}
        self.vehicle_history: dict[int, deque[tuple[int, tuple[int, int]]]] = {}
        self.vehicle_last_frame: dict[int, int] = {}
        self.episode_active = False
        self.frames_without_pair = 0

    @staticmethod
    def bottom_center(track: TrackedObject) -> tuple[int, int]:
        x1, _, x2, y2 = track.bbox
        return ((x1 + x2) // 2, y2)

    def _inside(self, track: TrackedObject) -> bool:
        x, y = self.bottom_center(track)
        return cv2.pointPolygonTest(
            self.polygon_pixels, (float(x), float(y)), False
        ) >= 0

    def evaluate(
        self,
        tracked_objects: list[TrackedObject],
        timestamp: float | None = None,
    ) -> SharedAisleWarningResult | None:
        """Receber todos os tracks de UM frame e devolver no máximo um alerta."""
        self.frame_index += 1
        frame_index = self.frame_index

        people = [
            t for t in tracked_objects
            if t.class_name == "person" and self._inside(t)
        ]
        vehicles = [
            t for t in tracked_objects
            if t.class_name == "industrial_vehicle" and self._inside(t)
        ]

        # Históricos de movimento não atravessam falhas na identidade do track.
        for vehicle in vehicles:
            vehicle_id = vehicle.track_id
            if self.vehicle_last_frame.get(vehicle_id) != frame_index - 1:
                self.vehicle_history[vehicle_id] = deque(
                    maxlen=self.motion_window_frames
                )
            self.vehicle_history[vehicle_id].append(
                (frame_index, self.bottom_center(vehicle))
            )
            self.vehicle_last_frame[vehicle_id] = frame_index

        # Não acumular históricos de IDs que saíram da zona/ficaram ausentes.
        # Se reaparecerem, a observação seguinte inicia uma janela nova.
        current_vehicle_ids = {v.track_id for v in vehicles}
        for previous_id in list(self.vehicle_history):
            if previous_id not in current_vehicle_ids:
                del self.vehicle_history[previous_id]
                self.vehicle_last_frame.pop(previous_id, None)

        current_pairs: dict[tuple[int, int], int] = {}
        candidates = []
        for person, vehicle in product(people, vehicles):
            pair = (person.track_id, vehicle.track_id)
            consecutive = self.pair_sequences.get(pair, 0) + 1
            current_pairs[pair] = consecutive

            positions = self.vehicle_history[vehicle.track_id]
            displacement = 0.0
            if len(positions) >= 2:
                start = positions[0][1]
                end = positions[-1][1]
                displacement = hypot(end[0] - start[0], end[1] - start[1])
            candidates.append((person, vehicle, consecutive, displacement))

        # Uma ausência de pares interrompe a confirmação; após ausência
        # sustentada, permite um episódio novo. Breves falhas não rearmam.
        self.pair_sequences = current_pairs
        if not candidates:
            self.frames_without_pair += 1
            if self.frames_without_pair >= self.rearm_after_frames:
                self.episode_active = False
            return None
        self.frames_without_pair = 0

        if self.episode_active:
            return None

        for person, vehicle, consecutive, displacement in candidates:
            if (
                consecutive >= self.confirmation_frames
                and displacement >= self.min_vehicle_displacement_px
            ):
                self.episode_active = True
                return SharedAisleWarningResult(
                    frame_index=frame_index,
                    source_time_seconds=timestamp,
                    person_track_id=person.track_id,
                    vehicle_track_id=vehicle.track_id,
                    confidence=min(person.confidence, vehicle.confidence),
                    person_bottom_center=self.bottom_center(person),
                    vehicle_bottom_center=self.bottom_center(vehicle),
                    consecutive_frames=consecutive,
                    vehicle_image_displacement_px=round(displacement, 2),
                )
        return None
