#include "readywidget.h"
#include "ui_readywidget.h"

readyWidget::readyWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::readyWidget)
{
    ui->setupUi(this);
}

readyWidget::~readyWidget()
{
    delete ui;
}
