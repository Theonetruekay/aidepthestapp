import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class DepthEstimationGUI extends JFrame {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private final VideoCapture capture;
    private final Mat frame;
    private final JLabel imageView;

    public DepthEstimationGUI() {
        super("Depth Estimation Application");
        
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLayout(new FlowLayout());

        imageView = new JLabel();
        add(imageView);

        capture = new VideoCapture();
        frame = new Mat();

        JButton startButton = new JButton("Start");
        startButton.addActionListener(e -> startCamera());
        add(startButton);

        JButton stopButton = new JButton("Stop");
        stopButton.addActionListener(e -> stopCamera());
        add(stopButton);

        setVisible(true);
    }

    private void startCamera() {
        capture.open(0); // Use 0 for default camera
        if (!capture.isOpened()) {
            System.out.println("Error: Camera not accessible");
            return;
        }
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280); // Set camera resolution
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);

        Runnable frameGrabber = new Runnable() {
            @Override
            public void run() {
                while (capture.isOpened()) {
                    capture.read(frame);
                    if (!frame.empty()) {
                        updateImageView(convertMatToImage(frame));
                    }
                }
            }
        };
        new Thread(frameGrabber).start();
    }

    private void stopCamera() {
        if (capture.isOpened()) {
            capture.release();
        }
    }

    private void updateImageView(Image image) {
        ImageIcon imageIcon = new ImageIcon(image);
        imageView.setIcon(imageIcon);
        pack();
    }

    private static Image convertMatToImage(Mat mat) {
        byte[] data = new byte[mat.width() * mat.height() * (int) mat.elemSize()];
        mat.get(0, 0, data);
        BufferedImage image = new BufferedImage(mat.width(), mat.height(), BufferedImage.TYPE_3BYTE_BGR);
        image.getRaster().setDataElements(0, 0, mat.width(), mat.height(), data);
        return image;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new DepthEstimationGUI();
            }
        });
    }
}
