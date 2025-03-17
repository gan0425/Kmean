#include "processimagecg.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ProcessImageCG w;
    w.show();
    return a.exec();
}
