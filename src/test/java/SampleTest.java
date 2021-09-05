import com.jml.linear_models.LinearRegression;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SampleTest { // Class must be public and follow the naming convention "*Test"
    LinearRegression linreg;


    @BeforeEach // Runs before each test
    void setUp() {
        linreg = new LinearRegression();
    }

    @Test // Defines a test method
    @DisplayName("Checking load model.") // define the name of the test which is displayed to the user
    public void demoTestCase() {

        // These are assertion statements which validates that expected and actual value is the same.
        // If they are not, the message at the end of the method is shown.
        assertNull(linreg.loadModel("file.mdl"), "LinearRegression.loadModel did not return null.");
        assertTrue(true, "Value should be true.");
    }

    @RepeatedTest(5) // Defines this test method should be executed multiple times. In this case 5.
    @DisplayName("Ensure correct handling of zero")
    public void demo2Test() {
        assertEquals(0, 0, "Multiple with zero should be zero");
        assertEquals(0, 0, "Multiple with zero should be zero");
    }
}
