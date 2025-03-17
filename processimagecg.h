#ifndef PROCESSIMAGECG_H
#define PROCESSIMAGECG_H

#include <QMainWindow>

#include "CGProcessImage.h"
#include <QFile>
#include <QString>

QT_BEGIN_NAMESPACE
namespace Ui { class ProcessImageCG; }
QT_END_NAMESPACE

class ProcessImageCG : public QMainWindow
{
    Q_OBJECT

public:
    ProcessImageCG(QWidget *parent = nullptr);
    ~ProcessImageCG();

private:
    Ui::ProcessImageCG *ui;

    CGProcessImage* processCG;
    bool SaveImage(QString _filepath, double**& _data, int _total_row, int _total_col, int maxValue);

};
#endif // PROCESSIMAGECG_H
