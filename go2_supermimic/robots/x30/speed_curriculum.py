"""Second X30 course: wider lateral and in-place yaw targets."""

from go2_supermimic.robots.x30.curriculum import Curriculum, LEVELS


SPEED_LEVELS = LEVELS | {
    'lateral': (.2, .3, .5, .75, 1., 1.25, 1.5, 1.75, 2.),
    'yaw': (.2, .4, .7, 1., 1.2, 1.5, 1.8, 2.),
}
SPEED_MODE_PROBABILITIES = (.10, .20, .10, .20, .15, .15, .10)


class SpeedCurriculum(Curriculum):
    levels = SPEED_LEVELS
    version = 2
    mode_probabilities = SPEED_MODE_PROBABILITIES
    probe_weight_defaults = Curriculum.probe_weight_defaults | {'lateral': .3, 'yaw': .3}
    probe_weight_max = .3

    @classmethod
    def from_prior(cls, prior):
        old = Curriculum()
        old.load_state_dict(prior)
        course = cls()
        for axis in course.levels:
            course.state['axes'][axis]['level'] = old.state['axes'][axis]['level']
        return course
