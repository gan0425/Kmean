#include "processimagecg.h"
#include "ui_processimagecg.h"

ProcessImageCG::ProcessImageCG(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ProcessImageCG)
{
    ui->setupUi(this);

    processCG = new CGProcessImage();
    double** rawData = nullptr;
    int row = 0, column = 0;
    CGProcessImage::ReadText("raw.txt", rawData, row, column);

    processCG->Process(rawData, row, column);

    double** calibData = nullptr;
    int calibRow = 0, calibColumn = 0;
    processCG->GetCaliData(calibData, calibRow, calibColumn);
    SaveImage("calib", calibData, calibRow, calibColumn, 65335);
}

ProcessImageCG::~ProcessImageCG()
{
    delete ui;




}

bool ProcessImageCG::SaveImage(QString _filepath, double**& _data, int _total_row, int _total_col, int maxValue) {

    QFile file(_filepath + ".png");
    file.open(QIODevice::WriteOnly);
    QImage m_Image_Scanline = QImage(_total_col, _total_row, QImage::Format::Format_RGBA64_Premultiplied);
    for (int row = 0; row < _total_row; row++) {
        for (int col = 0; col < _total_col; col++) {
            uint16_t value = static_cast<uint16_t>(_data[row][col]);
            //for (int tcol = 0; tcol < total_row; tcol++) {
            m_Image_Scanline.setPixelColor(QPoint(col, row), QColor(QRgba64::fromRgba64(value, value, value, maxValue)));
            //}
        }
    }
    return m_Image_Scanline.save(&file, "PNG", 100);

}
