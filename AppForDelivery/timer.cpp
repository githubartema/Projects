#include"timer.h"

class Timer : public QObject {
  Q_OBJECT
public:
  QReplyTimeout(QNetworkReply* reply, const int timeout) : QObject(reply) {
    Q_ASSERT(reply);
    if (reply) {
      QTimer::singleShot(timeout, this, SLOT(timeout()));
    }
  }

private slots:
  void timeout() {
    QNetworkReply* reply = static_cast<QNetworkReply*>(parent());
    if (reply->isRunning()) {
      reply->close();
    }
  }
};
