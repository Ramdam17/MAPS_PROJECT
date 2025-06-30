/**
 * Vec2f class - Represents a 2D vector with x and y components.
 */
class Vec2f {
    constructor(x, y) {
        // Use Float32Array as backing storage for better performance
        this.data = new Float32Array(2);
        this.data[0] = x;
        this.data[1] = y;
    }
    x() {
        return this.data[0];
    }
    y() {
        return this.data[1];
    }
    setX(x) {
        this.data[0] = x;
    }
    setY(y) {
        this.data[1] = y;
    }
    /** Returns a new vector that is the sum of this vector and v. */
    add(v) {
        return new Vec2f(this.x() + v.x(), this.y() + v.y());
    }
    /** Returns a new vector that is this vector minus v. */
    sub(v) {
        return new Vec2f(this.x() - v.x(), this.y() - v.y());
    }
    /** Returns a new vector that is this vector multiplied by scalar s. */
    mul(s) {
        return new Vec2f(this.x() * s, this.y() * s);
    }
    /** Returns a new vector that is this vector divided by scalar s. */
    div(s) {
        return new Vec2f(this.x() / s, this.y() / s);
    }
    /** Returns the length (magnitude) of this vector. */
    length() {
        return Math.sqrt(this.x() * this.x() + this.y() * this.y());
    }
    /** Returns a new vector that is this vector normalized (length = 1). */
    normalize() {
        const length = this.length();
        return new Vec2f(this.x() / length, this.y() / length);
    }
}
/**
 * Mat3f class - Represents a 3x3 matrix for 2D transformations.
 * Matrix is organized in row-major order:
 * | 0 1 2 |
 * | 3 4 5 |
 * | 6 7 8 |
 */
class Mat3f {
    constructor(a, b, c, d, e, f, g, h, i) {
        // Use Float32Array as backing storage for better performance
        this.data = new Float32Array(9);
        this.data[0] = a;
        this.data[1] = b;
        this.data[2] = c;
        this.data[3] = d;
        this.data[4] = e;
        this.data[5] = f;
        this.data[6] = g;
        this.data[7] = h;
        this.data[8] = i;
    }
    /** Row-column access - get element at row r, column c (both 0-based) */
    get(r, c) {
        return this.data[r * 3 + c];
    }
    /** Set element at row r, column c (both 0-based) to value */
    set(r, c, value) {
        this.data[r * 3 + c] = value;
        return this;
    }
    /** Returns an identity matrix (no transformation). */
    static identity() {
        return new Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
    /** Returns a translation matrix for moving by (x, y). */
    static translate(x, y) {
        return new Mat3f(1, 0, x, 0, 1, y, 0, 0, 1);
    }
    /** Returns a scaling matrix for scaling by factors (x, y). */
    static scale(x, y) {
        return new Mat3f(x, 0, 0, 0, y, 0, 0, 0, 1);
    }
    /** Returns a rotation matrix for rotating by angle (in radians). */
    static rotate(angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return new Mat3f(c, s, 0, -s, c, 0, 0, 0, 1);
    }
    /**
     * Transforms a vector v using this matrix and returns the result.
     * Converts the 2D vector to homogeneous coordinates for the transformation.
     */
    transform(v) {
        return new Vec2f(this.get(0, 0) * v.x() + this.get(0, 1) * v.y() + this.get(0, 2), this.get(1, 0) * v.x() + this.get(1, 1) * v.y() + this.get(1, 2));
    }
    /** Returns a new matrix that is the result of multiplying this matrix by m. */
    mul(m) {
        return new Mat3f(this.get(0, 0) * m.get(0, 0) + this.get(0, 1) * m.get(1, 0) + this.get(0, 2) * m.get(2, 0), this.get(0, 0) * m.get(0, 1) + this.get(0, 1) * m.get(1, 1) + this.get(0, 2) * m.get(2, 1), this.get(0, 0) * m.get(0, 2) + this.get(0, 1) * m.get(1, 2) + this.get(0, 2) * m.get(2, 2), this.get(1, 0) * m.get(0, 0) + this.get(1, 1) * m.get(1, 0) + this.get(1, 2) * m.get(2, 0), this.get(1, 0) * m.get(0, 1) + this.get(1, 1) * m.get(1, 1) + this.get(1, 2) * m.get(2, 1), this.get(1, 0) * m.get(0, 2) + this.get(1, 1) * m.get(1, 2) + this.get(1, 2) * m.get(2, 2), this.get(2, 0) * m.get(0, 0) + this.get(2, 1) * m.get(1, 0) + this.get(2, 2) * m.get(2, 0), this.get(2, 0) * m.get(0, 1) + this.get(2, 1) * m.get(1, 1) + this.get(2, 2) * m.get(2, 1), this.get(2, 0) * m.get(0, 2) + this.get(2, 1) * m.get(1, 2) + this.get(2, 2) * m.get(2, 2));
    }
    /** Returns a new matrix that is the inverse of this matrix. */
    inverse() {
        const det = this.get(0, 0) * (this.get(1, 1) * this.get(2, 2) - this.get(1, 2) * this.get(2, 1)) -
            this.get(0, 1) * (this.get(1, 0) * this.get(2, 2) - this.get(1, 2) * this.get(2, 0)) +
            this.get(0, 2) * (this.get(1, 0) * this.get(2, 1) - this.get(1, 1) * this.get(2, 0));
        if (det == 0) {
            throw new Error('Matrix is not invertible');
        }
        return new Mat3f((this.get(1, 1) * this.get(2, 2) - this.get(1, 2) * this.get(2, 1)) / det, (this.get(0, 2) * this.get(2, 1) - this.get(0, 1) * this.get(2, 2)) / det, (this.get(0, 1) * this.get(1, 2) - this.get(0, 2) * this.get(1, 1)) / det, (this.get(1, 2) * this.get(2, 0) - this.get(1, 0) * this.get(2, 2)) / det, (this.get(0, 0) * this.get(2, 2) - this.get(0, 2) * this.get(2, 0)) / det, (this.get(0, 2) * this.get(1, 0) - this.get(0, 0) * this.get(1, 2)) / det, (this.get(1, 0) * this.get(2, 2) - this.get(1, 2) * this.get(2, 0)) / det, (this.get(0, 1) * this.get(2, 0) - this.get(0, 0) * this.get(2, 1)) / det, (this.get(0, 0) * this.get(1, 1) - this.get(0, 1) * this.get(1, 0)) / det);
    }
}
export { Vec2f, Mat3f };
