#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "readyscene.h"
#include "jsondownloader.h"
#include <QNetworkRequest>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QTimer>
#include <QSettings>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_enterButton_clicked();
    void onStart();    
    void on_suggestButton_clicked();

    void on_recallButton_clicked();

    void recallingScene(bool f);
    void enterScene(bool f);
public:
    QSettings settings;

private:
    Ui::MainWindow *ui;
    QNetworkAccessManager *manager;
    QNetworkRequest request;
    QString answer;
    readyScene * secondScene = new readyScene;
    QScreen *screen;
    bool entered = false;
};

#endif // MAINWINDOW_H


