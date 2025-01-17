from qstrader.broker.transaction.transaction import Transaction


class OrderMemo(object):
    def __init__(self, path="memo.csv"):
        self.file = open(path, 'w')

    def log_transaction(self, txn: Transaction):

        line = ",".join([str(txn.dt), txn.asset, str(txn.quantity), str(txn.price), str(txn.adjust_price)])
        self.file.write(line)
        self.file.write("\r\n")

    def close(self):
        self.file.close()
