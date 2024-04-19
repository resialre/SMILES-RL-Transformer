import torch
import torch.nn.functional as F
import torch.optim as optim
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelDistillation:
    def __init__(self, optimizer, alpha, batch_size, n_steps, temp):
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.temp = temp
        self.optimizer = optimizer

    def transfer(self, student, teacher):
        self.loss_history = [] 
        self.min_loss_history = []
        self.max_loss_history = []
        for step in range(self.n_steps):
            teacher_seq, teacher_logits = teacher.teach(self.batch_size)
            student_logits = student.q_values(teacher_seq)[:,:,:]

            teacher_soft = F.softmax(teacher_logits / self.temp, dim=-1)
            student_soft = F.log_softmax(student_logits / self.temp, dim=-1)
            flat_target = teacher_seq[:, 1:].reshape(-1)  
            flat_logits = student_logits.reshape(-1, student_logits.size(-1))  

            soft_target_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temp ** 2)
            label_loss = F.cross_entropy(flat_logits, flat_target)

            loss = self.alpha * soft_target_loss + (1 - self.alpha) * label_loss
            if loss.dim() > 0:
                loss = loss.mean()

            if step % 10 == 0:
                print(f"Step: {step}")
                print(f"Teacher sequence: {teacher_seq[:1,1:]}")
                print(f"Teacher logit: {torch.argmax(teacher_logits,dim=-1)[:1,:]}")
                print(f"Student sequence: {torch.argmax(student_logits,dim=-1)[:1,:]}")
                print(f"Step {step}/{self.n_steps}, Loss: {loss.item()}")

            self.loss_history.append(loss.item())
            if len(self.min_loss_history) == 0 or loss.item() < self.min_loss_history[-1]:
                self.min_loss_history.append(loss.item())
            else:
                self.min_loss_history.append(self.min_loss_history[-1])

            if len(self.max_loss_history) == 0 or loss.item() > self.max_loss_history[-1]:
                self.max_loss_history.append(loss.item())
            else:
                self.max_loss_history.append(self.max_loss_history[-1])

            self.update_params(loss)

    def update_params(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_losses_to_csv(self, filename='loss_data.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'Loss', 'Min Loss', 'Max Loss'])
            for step, (loss, min_loss, max_loss) in enumerate(zip(self.loss_history, self.min_loss_history, self.max_loss_history)):
                writer.writerow([step, loss, min_loss, max_loss])

