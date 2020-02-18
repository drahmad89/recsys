import numpy as np

class CheckPoints():
    def __init__(self, input_file, model_name, loss, lr, mode,
                 direc="./checkpoints/"):
        self.input_file = input_file
        self.model_name = model_name
        self.loss = loss
        self.lr = lr
        self.direc = direc
        self.file_writier = None
        self.mode = mode

    def create_file_writer(self):
        try:
            self.file_writer =  open("{}{}/{}/{}/{}_{}_{}_{}.txt".format(self.direc, self.input_file,
                                                            self.loss,
                                                            self.mode,
                                                            self.input_file,
                                                            self.mode,
                                                            self.loss,self.lr),
                                                             "w")
        except:
            print("Could not create file writer")

        return
    def write_line(self, iteration, epoch, loss, acc):
        self.file_writer.write("{} {} {} {}\n".format(iteration, epoch, loss,
                                                acc))

