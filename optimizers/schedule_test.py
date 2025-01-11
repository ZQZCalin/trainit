"""Tests learning rate schedules."""

from optimizers.schedule import warmup_const_linear_decay_schedule


def test_scheduler():
    return


def test_warmup_const_linear_decay_schedule():
    warmup = [0, 10]
    const = [0, 10]
    total = 30
    peak = 100.0
    for w in warmup:
        for c in const:
            print(f"\n>> Testing schedule for warmup={w}, const={c}, total={total}, peak={peak}")
            lr = warmup_const_linear_decay_schedule(
                peak, w, c, total
            )
            for t in range(total+2):
                print(f"iter {t}: lr = {lr(t)}")


if __name__ == "__main__":
    test_scheduler()