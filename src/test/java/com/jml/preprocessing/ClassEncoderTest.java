package com.jml.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ClassEncoderTest {

    ClassEncoder enc;

    @Test
    void exceptionTest() {
        enc = new ClassEncoder();

        assertThrows(IllegalStateException.class, () -> enc.encode(new String[][]{{"cat"}}));
        assertThrows(IllegalStateException.class, () -> enc.decode(new int[][]{{0}}));
    }


    @Test
    void ignoreTest() {
        enc = new ClassEncoder(true);

        String[][] targets = {{"cat"}, {"dog"}, {"cat"}, {"cat"}, {"panda"}, {"penguin"}, {"cat"}, {"cat"}, {"dog"}};
        int[][] expTargetEnc = {{0}, {1}, {0}, {0}, {2}, {3}, {0}, {0}, {1}};

        String[][] val = {{"cat"}, {"hamster"}, {"human"}, {"dog"}, {"panda"}};
        int[][] expValEnc = {{0}, {-1}, {-1}, {1}, {2}};
        String[][] expValDec = {{"cat"}, {null}, {null}, {"dog"}, {"panda"}};

        enc.fit(targets);

        int[][] targetEnc = enc.encode(targets);
        int[][] valEnc = enc.encode(val);

        String[][] targetDec = enc.decode(expTargetEnc);
        String[][] valDec = enc.decode(expValEnc);

        assertArrayEquals(expTargetEnc, targetEnc);
        assertArrayEquals(expValEnc, valEnc);
        assertArrayEquals(expValDec, valDec);
        assertArrayEquals(targets, targetDec);
    }


    @Test
    void dontIgnoreTest() {
        enc = new ClassEncoder();

        String[][] targets = {{"cat"}, {"dog"}, {"cat"}, {"cat"}, {"panda"}, {"penguin"}, {"cat"}, {"cat"}, {"dog"}};
        int[][] expTargetEnc = {{0}, {1}, {0}, {0}, {2}, {3}, {0}, {0}, {1}};

        String[][] val = {{"cat"}, {"dog"}, {"panda"}};
        int[][] expValEnc = {{0}, {1}, {2}};
        String[][] expValDec = {{"cat"}, {"dog"}, {"panda"}};

        enc.fit(targets);

        int[][] targetEnc = enc.encode(targets);
        int[][] valEnc = enc.encode(val);

        String[][] targetDec = enc.decode(expTargetEnc);
        String[][] valDec = enc.decode(expValEnc);

        assertArrayEquals(expTargetEnc, targetEnc);
        assertArrayEquals(expValEnc, valEnc);
        assertArrayEquals(expValDec, valDec);
        assertArrayEquals(targets, targetDec);


        String[][] unseen = {{"hamster"}, {"cat"}, {"dog"}};
        assertThrows(IllegalStateException.class, () -> enc.encode(unseen));
    }
}
