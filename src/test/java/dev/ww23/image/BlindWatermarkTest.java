/*
 * Copyright (c) 2020 ww23(https://github.com/ww23/BlindWatermark).
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dev.ww23.image;

import dev.ww23.image.converter.Converter;
import dev.ww23.image.converter.DctConverter;
import dev.ww23.image.converter.DftConverter;
import dev.ww23.image.dencoder.Decoder;
import dev.ww23.image.dencoder.Encoder;
import dev.ww23.image.dencoder.ImageEncoder;
import dev.ww23.image.dencoder.TextEncoder;
import dev.ww23.image.util.Utils;
import dev.ww23.image.util.WatermarkUtil;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_8U;

class BlindWatermarkTest {

    private static final String SRC = "image/gakki-src.png";
    private static final String TEXT_WM = "X123456789";
    private static final String IMAGE_WM = "image/watermark.png";

    static {
        Loader.load(opencv_java.class);
    }

    @Test
    void dctImage() {
        Converter converter = new DctConverter();
        Encoder encoder = new ImageEncoder(converter);
        Decoder decoder = new Decoder(converter);
        encoder.encode(SRC, IMAGE_WM, "image/gakki-dct-img-ec.jpg");
        decoder.decode("image/gakki-dct-img-ec.jpg", "image/gakki-dct-img-dc.jpg");
    }

    @Test
    void dctText() {
        Converter converter = new DctConverter();
        Encoder encoder = new TextEncoder(converter);
        Decoder decoder = new Decoder(converter);
        encoder.encode(SRC, TEXT_WM, "image/gakki-dct-text-ec.jpg");
        decoder.decode("image/gakki-dct-text-ec.jpg", "image/gakki-dct-text-dc.jpg");
    }

    int threshold = 240;

    @Test
    void dctTextA() {
        Converter converter = new DctConverter();
        Encoder encoder = new TextEncoder(converter);
        Decoder decoder = new Decoder(converter);
        encoder.encode("image/white-src.jpg", TEXT_WM, "image/white-src-dct-ec.jpg");
        incPixes("image/white-src-dct-ec.jpg", "image/white-src-dct-ec-inc.jpg");
        exchangePix("image/white-src-dct-ec-inc.jpg", "image/white-src-dct-ec-spread.jpg");
        exchangePix9("image/white-src-dct-ec-spread.jpg", "image/white-src-dct-ec-spread.jpg");
        exchangePix("image/white-src-dct-ec-spread.jpg", "image/white-src-dct-ec-spread.jpg");
        exchangePix("image/white-src-dct-ec-spread.jpg", "image/white-src-dct-ec-spread-back.jpg");
        exchangePix9("image/white-src-dct-ec-spread-back.jpg", "image/white-src-dct-ec-spread-back.jpg");
        exchangePix("image/white-src-dct-ec-spread-back.jpg", "image/white-src-dct-ec-spread-back.jpg");
        processTransparentPng("image/white-src-dct-ec-spread.jpg", "image/white-src-dct-ec-alpha.png");
//        Mat mat = incPixesBack("image/white-src-dct-ec-spread-back.jpg");
//        decoder.decode(mat, "image/white-src-dct-dc.jpg");
        decoder.decode("image/white-src-dct-ec-spread-back.jpg", "image/white-src-dct-dc.jpg");
    }

    private Mat incPixesBack(String img) {
        Mat mat = Utils.read(img, CV_8U);
        int cols = mat.cols();
        int rows = mat.rows();
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                double[] pixes = mat.get(j, i);
                if (pixes[0] < threshold) {
                    mat.put(j, i, pixes[0] * 5 - threshold * 4);
                }
            }
        }
        return mat;
    }

    private void incPixes(String in, String out) {
        Mat ecMat = Utils.read(in, CV_8U);
        List<Mat> matList = new ArrayList<>(3);
        Core.split(ecMat, matList);
        int cols = ecMat.cols();
        int rows = ecMat.rows();
        Mat mat = matList.get(0);
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                double[] pixes = mat.get(j, i);
                if (pixes[0] < threshold) {
                    mat.put(j, i, pixes[0] + (threshold - pixes[0]) * 4 / 5);
                }
            }
        }
        Core.merge(matList, ecMat);
        Imgcodecs.imwrite(out, ecMat);
    }

    private void exchangePix(String in, String out) {
        Mat mat = Utils.read(in, CV_8U);
        int cols = mat.cols();
        int rows = mat.rows();
        int cellCols = cols / 3;
        int cellRows = rows / 3;
        int num = 8;
        for (int i = 0; i < cellCols; i += 2) {
            // 内圈三个
            for (int j = 0; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j + cellRows, i + cellCols);
                mat.put(j, i, pixes2);
                mat.put(j + cellRows, i + cellCols, pixes1);
            }
            for (int j = 1; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j, i + cellCols);
                mat.put(j, i, pixes2);
                mat.put(j, i + cellCols, pixes1);
            }
            for (int j = 2; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j + cellRows, i);
                mat.put(j, i, pixes2);
                mat.put(j + cellRows, i, pixes1);
            }
            //外圈五个
            for (int j = 3; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j + cellRows * 2, i + cellCols * 2);
                mat.put(j, i, pixes2);
                mat.put(j + cellRows * 2, i + cellCols * 2, pixes1);
            }
            for (int j = 4; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j, i + cellCols * 2);
                mat.put(j, i, pixes2);
                mat.put(j, i + cellCols * 2, pixes1);
            }
            for (int j = 5; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j + cellRows * 2, i);
                mat.put(j, i, pixes2);
                mat.put(j + cellRows * 2, i, pixes1);
            }
            for (int j = 6; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j + cellRows, i + cellCols * 2);
                mat.put(j, i, pixes2);
                mat.put(j + cellRows, i + cellCols * 2, pixes1);
            }
            for (int j = 7; j < cellRows; j += num) {
                double[] pixes1 = mat.get(j, i);
                double[] pixes2 = mat.get(j + cellRows * 2, i + cellCols);
                mat.put(j, i, pixes2);
                mat.put(j + cellRows * 2, i + cellCols, pixes1);
            }
        }
        Imgcodecs.imwrite(out, mat);
    }


    private void exchangePix9(String in, String out) {
        Mat mat = Utils.read(in, CV_8U);
        int cols = mat.cols();
        int rows = mat.rows();
        int cellCols = cols / 3;
        int cellRows = rows / 3;
        int num = 9;
        // 内圈三个
        for (int row = 0; row < cellRows; ++row) {
            for (int col = 0; col < cellCols; ++col) {
                int exchangeRow = 0;
                int exchangeCol = 0;
                int remainder = (col + row) % num;
                if (remainder == 0) {
                    exchangeRow = row;
                    exchangeCol = col + cellCols;
                } else if (remainder == 1) {
                    exchangeRow = row + cellRows;
                    exchangeCol = col + cellCols;
                } else if (remainder == 2) {
                    exchangeRow = row + cellRows;
                    exchangeCol = col;
                } else if (remainder == 3) {
                    exchangeRow = row;
                    exchangeCol = col + cellCols * 2;
                } else if (remainder == 4) {
                    exchangeRow = row + cellRows;
                    exchangeCol = col + cellCols * 2;
                } else if (remainder == 5) {
                    exchangeRow = row + cellRows * 2;
                    exchangeCol = col + cellCols * 2;
                } else if (remainder == 6) {
                    exchangeRow = row + cellRows * 2;
                    exchangeCol = col + cellCols;
                } else if (remainder == 7) {
                    exchangeRow = row + cellRows * 2;
                    exchangeCol = col;
                } else if (remainder == 8) {
                }
                if (exchangeRow > 0 && exchangeCol > 0) {
                    double[] pixes1 = mat.get(row, col);
                    double[] pixes2 = mat.get(exchangeRow, exchangeCol);
                    mat.put(row, col, pixes2);
                    mat.put(exchangeRow, exchangeCol, pixes1);
                }
            }
        }
        Imgcodecs.imwrite(out, mat);
    }

    private void exchangePix2(String in, String out) {
        Mat mat = Utils.read(in, CV_8U);
        int cols = mat.cols();
        int rows = mat.rows();
        int cellCols = cols / 2;
        int cellRows = rows / 2;
        for (int col = 0; col < cellCols; ++col) {
            // 内圈三个
            for (int row = 0; row < cellRows; ++row) {
                int exchangeRow = 0;
                int exchangeCol = 0;
                if ((col + row) % 4 == 0) {
                    exchangeRow = row;
                    exchangeCol = col + cellCols;
                } else if ((col + row) % 4 == 1) {
                    exchangeRow = row + cellRows;
                    exchangeCol = col + cellCols;
                } else if ((col + row) % 4 == 2) {
                    exchangeRow = row + cellRows;
                    exchangeCol = col;
                } else if ((col + row) % 4 == 3) {
                }
                if (exchangeRow > 0 && exchangeCol > 0) {
                    double[] pixes1 = mat.get(row, col);
                    double[] pixes2 = mat.get(exchangeRow, exchangeCol);
                    mat.put(row, col, pixes2);
                    mat.put(exchangeRow, exchangeCol, pixes1);
                }
            }
        }
        Imgcodecs.imwrite(out, mat);
    }

    // 透明png
    private Mat processTransparentPng(String in, String out) {
        Mat ecMat = Utils.read(in, CV_8U);
        List<Mat> matList = new ArrayList<>(3);
        Core.split(ecMat, matList);
        int cols = ecMat.cols();
        int rows = ecMat.rows();
        // 分成四象限"田"
        int splitCols = cols / 2;
        int spitRows = rows / 2;
        List<Mat> pngMatList = new ArrayList<>(3);
        pngMatList.add(matList.get(0));
        pngMatList.add(matList.get(0));
        pngMatList.add(matList.get(0));
        Mat zeros = Mat.zeros(rows, cols, CV_8U);
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                double[] pixes = matList.get(0).get(j, i);
                if (pixes[0] == 255) {
                    zeros.put(j, i, 0);
                } else {
                    zeros.put(j, i, 255);
                }
            }
        }
        pngMatList.add(zeros);
        Core.merge(pngMatList, ecMat);
        Imgcodecs.imwrite(out, ecMat);
        return ecMat;
    }

    @Test
    void dctDecode() {
        Converter converter = new DctConverter();
        Decoder decoder = new Decoder(converter);
        decoder.decode("image/1111.png", "image/1111-dc.png");
    }

    @Test
    void dftImage() {
        Converter converter = new DftConverter();
        Encoder encoder = new ImageEncoder(converter);
        encoder.encode(SRC, IMAGE_WM, "image/gakki-dft-img-ec.png");
        Decoder decoder = new Decoder(converter);
        decoder.decode("image/gakki-dft-img-ec.png", "image/gakki-dft-img-dc.png");
    }

    @Test
    void dftText() {
        Converter converter = new DftConverter();
        Encoder encoder = new TextEncoder(converter);
        Decoder decoder = new Decoder(converter);
        encoder.encode("image/white-src.png", TEXT_WM, "image/white-src-dft-ec.jpg");
        decoder.decode("image/white-src-dft-ec.jpg", "image/white-src-dft-dc.jpg");
    }

    @Test
    void dftTextMy() {
        Mat src = Utils.read("image/white-src.jpg", CvType.CV_8S);
        Mat addWatermarkMat = WatermarkUtil.addImageWatermarkWithText(src, "X123456789");
        Imgcodecs.imwrite("image/white-src-dft-ec.jpg", addWatermarkMat);
        Mat watermarkMat = WatermarkUtil.getImageWatermarkWithText(addWatermarkMat);
        Imgcodecs.imwrite("image/white-src-dft-dc.jpg", watermarkMat);
    }

}
