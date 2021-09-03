import linalg.Matrix;
import linalg.Decompose;

public class Test {
    public static void main(String[] args) {
        double[][] a = {{1, 2, 3},
                        {4, 5, 6}};
        double[][] b = {{1, 2},
                        {3, 4},
                        {5, 6}};
        Matrix A = new Matrix(a);
        Matrix B = new Matrix(b);

        System.out.println("A:\n" + A + "\n\n");
        System.out.println("B:\n" + B + "\n\n");
        System.out.println("A*B:\n" + A.mult(B) + "\n\n");

        System.out.println("----------------------------------------------" +
                "\nQR decomposition of B...\n");
        Matrix[] QR = Decompose.QR(B);
        System.out.println("Q:\n" + QR[0] + "\n\n");
        System.out.println("R:\n" + QR[1] + "\n");
        System.out.println("Q*R=A: " + QR[0].mult(QR[1]).round(10).equals(B));
        System.out.println("Q is orthogonal?: " + QR[0].isOrthogonal());
        System.out.println("R is upper-triangular?: " + QR[1].isTriU());

    }
}
