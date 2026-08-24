import unittest

from AttentionGuidance import FreeUParameters, FreeUSchedule


class _FakeUNet:
    def __init__(self):
        self.calls = []
        self.disabled = 0

    def enable_freeu(self, s1, s2, b1, b2):
        self.calls.append((s1, s2, b1, b2))

    def disable_freeu(self):
        self.disabled += 1


class FreeUTest(unittest.TestCase):
    def test_constant_schedule_is_reentrant_and_interpolates(self):
        schedule = FreeUSchedule.constant((0.6, 0.4, 1.1, 1.2))
        model = _FakeUNet()
        first = schedule.apply(model, 0, 3)
        last = schedule.apply(model, 2, 3)
        self.assertEqual(first.as_tuple(), last.as_tuple())
        self.assertEqual(len(model.calls), 2)
        schedule.disable(model)
        self.assertEqual(model.disabled, 1)

        dynamic = FreeUSchedule(
            (
                (0.0, (1.0, 1.0, 1.0, 1.0)),
                (1.0, (0.5, 0.25, 1.5, 2.0)),
            )
        )
        middle = dynamic.at(0.5)
        self.assertEqual(middle.as_tuple(), (0.75, 0.625, 1.25, 1.5))

    def test_schedule_requires_endpoints_and_validates_ranges(self):
        with self.assertRaises(ValueError):
            FreeUSchedule(((0.5, (1.0, 1.0, 1.0, 1.0)),))
        with self.assertRaises(ValueError):
            FreeUParameters.from_sequence((1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            FreeUParameters(3.0, 1.0, 1.0, 1.0)

    def test_step_bounds_are_checked(self):
        schedule = FreeUSchedule.constant((1.0, 1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            schedule.apply(_FakeUNet(), 3, 3)
        with self.assertRaises(ValueError):
            schedule.apply(_FakeUNet(), 0, 0)


if __name__ == "__main__":
    unittest.main()
