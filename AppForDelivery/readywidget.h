#ifndef READYWIDGET_H
#define READYWIDGET_H

#include <QWidget>

namespace Ui {
class readyWidget;
}

class readyWidget : public QWidget
{
    Q_OBJECT

public:
    explicit readyWidget(QWidget *parent = nullptr);
    ~readyWidget();

private:
    Ui::readyWidget *ui;
};

#endif // READYWIDGET_H
