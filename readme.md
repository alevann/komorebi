# Komorebi **WIP**

A garbage implementation of a 3D model viewer in Rust.

# Notes

Why not use loops in the Matrix and Vector implementation?
It turns out, not using loops is a lot faster.

Rough tests show a very clear difference in time when both a loopy and a loop-free implementation are timed.<br>
More specifically here's a few results:

* `Vector.mag` is about ~3 times faster without loops. (Loopy: ~154ms - Loop-Free: ~66ms)
* `Matrix.mul` is about ~5 times faster without loops. (Loopy: ~2.5s - Loop-free: ~420ms)

To run these tests, each function has been called 1_000_000 times with pre-loaded arguments.
