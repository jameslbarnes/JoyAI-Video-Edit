"""Bounded WebSocket dial writes, independent of the frame receive loop."""
import asyncio

from .shader_strength import validate_mode, validate_strength


class ShaderControlQueue:
    def __init__(self, strength, notify):
        self.strength, self.notify = strength, notify
        self.pending = {}  # at most one latest command for each of two controls
        self.task = None
        self.closed = False

    def submit(self, kind, payload):
        if self.closed:
            raise ValueError('Shader controls are closed')
        if kind not in ('set_strength', 'set_shader_overlay'):
            raise ValueError('Unknown shader control')
        command = {'value': validate_strength(payload.get('value'))}
        if kind == 'set_strength':
            if payload.get('mode') is not None:
                command['mode'] = validate_mode(payload['mode'])
            if payload.get('outputMix') is not None:
                command['output_mix'] = validate_strength(payload['outputMix'])
        self.pending.pop(kind, None)
        self.pending[kind] = (command, payload.get('requestId'))
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    async def _run(self):
        while self.pending and not self.closed:
            kind = next(iter(self.pending))
            command, request_id = self.pending.pop(kind)
            try:
                setter = self.strength.set if kind == 'set_strength' else self.strength.set_overlay
                state = await asyncio.to_thread(setter, **command)
                response = {'type': 'strength_queued', 'strength': state}
            except ValueError as exc:
                response = {'type': 'strength_error', 'message': str(exc)}
            except OSError:
                response = {'type': 'strength_error', 'message': 'Shader setting could not be saved; change was not applied'}
            response.update(control=kind, requestId=request_id)
            if not self.closed:
                try:
                    await self.notify(response)
                except Exception:
                    # Disconnected peer: finish ownership of the active write,
                    # discard unsent gestures, and let the socket owner tear down.
                    self.closed = True
                    self.pending.clear()

    async def close(self):
        self.closed = True
        self.pending.clear()
        if self.task is not None:
            await self.task  # cancellation cannot stop an in-flight disk write
