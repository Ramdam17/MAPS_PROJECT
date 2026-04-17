/**
 * A grid of booleans.
 * This is much faster than using a 2D array of booleans or hash tables.
 * It uses a 1D array of uint8 booleans, and the index is computed as y * width + x.
 */
export class Grid {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.data = new Uint8Array(width * height);
    }
    /** Sets the value of a cell. */
    set(x, y, value) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
            return;
        }
        this.data[y * this.width + x] = value ? 1 : 0;
    }
    /** Gets the value of a cell. */
    get(x, y) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
            return false;
        }
        return this.data[y * this.width + x] === 1;
    }
    /** Clears the grid. */
    clear() {
        this.data.fill(0);
    }
}
