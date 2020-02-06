#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QNetworkRequest>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QSslSocket>
#include <QLineEdit>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
      manager = new QNetworkAccessManager();
        QObject::connect(manager, &QNetworkAccessManager::finished,this, [=](QNetworkReply *reply) {
                if (reply->error()) {
                    qDebug() << reply->errorString();
                    return;
                }

                this->answer = reply->readAll();

                //qDebug() << reply->attribute(QNetworkRequest::HttpStatusCodeAttribute);
                qDebug() << answer;

                QJsonDocument jsonResponse = QJsonDocument::fromJson(answer.toUtf8());
                QJsonObject jsonObject = jsonResponse.object();
                QJsonObject a = jsonObject.value(QString("res")).toObject();
                qDebug() << a.value(QString("token"));

            }
        );
    }

    MainWindow::~MainWindow()
    {
        delete ui;
        delete manager;
    }


void MainWindow::on_enterButton_clicked()
{
    QString login = ui->loginField->text();
    QString password = ui->passwordField->text();

    request.setUrl(QUrl("http://api.torianik.online:5000/login?login=" + login + "&password=" + password));
    request.setSslConfiguration(QSslConfiguration::defaultConfiguration());
    manager->get(request);

}
