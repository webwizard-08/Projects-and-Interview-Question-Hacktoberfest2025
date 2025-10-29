// ExpenseTracker.js
// Works in both Browser (UI) and Node.js (Console)
// Tracks income/expenses, categories, and summary stats

class ExpenseTracker {
  constructor() {
    this.transactions = this.loadTransactions();
    this.calculateTotals();
  }

  // Load transactions from localStorage (browser) or memory (Node)
  loadTransactions() {
    if (typeof localStorage !== 'undefined') {
      return JSON.parse(localStorage.getItem('transactions')) || [];
    }
    return []; // Node.js fallback
  }

  // Save transactions
  saveTransactions() {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('transactions', JSON.stringify(this.transactions));
    }
  }

  // Calculate totals
  calculateTotals() {
    this.income = this.transactions
      .filter(t => t.type === 'income')
      .reduce((sum, t) => sum + t.amount, 0);

    this.expense = this.transactions
      .filter(t => t.type === 'expense')
      .reduce((sum, t) => sum + t.amount, 0);

    this.balance = this.income - this.expense;
  }

  addTransaction(description, amount, type, category) {
    const transaction = {
      id: Date.now(),
      description,
      amount: parseFloat(amount),
      type,
      category,
      date: new Date().toLocaleDateString()
    };
    this.transactions.push(transaction);
    this.calculateTotals();
    this.saveTransactions();
    return transaction;
  }

  deleteTransaction(id) {
    this.transactions = this.transactions.filter(t => t.id !== id);
    this.calculateTotals();
    this.saveTransactions();
  }

  getSummary() {
    return {
      balance: this.balance,
      income: this.income,
      expense: this.expense,
      totalTransactions: this.transactions.length
    };
  }

  getSpendingByCategory() {
    const spending = {};
    this.transactions
      .filter(t => t.type === 'expense')
      .forEach(t => {
        spending[t.category] = (spending[t.category] || 0) + t.amount;
      });
    return spending;
  }
}

// ==============================
// ===== Browser Environment ====
// ==============================
if (typeof document !== 'undefined') {
  const tracker = new ExpenseTracker();

  const app = document.createElement('div');
  app.innerHTML = `
    <h2>💰 Expense Tracker</h2>
    <form id="expenseForm">
      <input type="text" id="desc" placeholder="Description" required />
      <input type="number" id="amount" placeholder="Amount (₹)" required />
      <select id="type">
        <option value="expense">Expense</option>
        <option value="income">Income</option>
      </select>
      <input type="text" id="category" placeholder="Category" required />
      <button type="submit">Add</button>
    </form>
    <h3 id="summary"></h3>
    <ul id="list"></ul>
  `;

  document.body.appendChild(app);

  const form = document.getElementById('expenseForm');
  const summaryEl = document.getElementById('summary');
  const listEl = document.getElementById('list');

  function render() {
    const { balance, income, expense } = tracker.getSummary();
    summaryEl.textContent = `Balance: ₹${balance} | Income: ₹${income} | Expense: ₹${expense}`;
    listEl.innerHTML = '';
    tracker.transactions.forEach(t => {
      const li = document.createElement('li');
      li.textContent = `${t.description} — ₹${t.amount} (${t.type}, ${t.category})`;
      const del = document.createElement('button');
      del.textContent = '❌';
      del.onclick = () => {
        tracker.deleteTransaction(t.id);
        render();
      };
      li.appendChild(del);
      listEl.appendChild(li);
    });
  }

  form.onsubmit = e => {
    e.preventDefault();
    const desc = document.getElementById('desc').value.trim();
    const amount = parseFloat(document.getElementById('amount').value);
    const type = document.getElementById('type').value;
    const category = document.getElementById('category').value.trim();

    if (desc && amount > 0 && category) {
      tracker.addTransaction(desc, amount, type, category);
      form.reset();
      render();
    } else {
      alert('Please fill all fields correctly.');
    }
  };

  render();
}

// ==============================
// ===== Node.js Environment ====
// ==============================
if (typeof module !== 'undefined' && module.exports) {
  const tracker = new ExpenseTracker();

  // Sample transactions (for console run)
  tracker.addTransaction('Salary', 5000, 'income', 'Work');
  tracker.addTransaction('Groceries', 300, 'expense', 'Food');
  tracker.addTransaction('Electricity', 120, 'expense', 'Utilities');
  tracker.addTransaction('Freelance', 800, 'income', 'Work');

  const summary = tracker.getSummary();

  console.log('=== EXPENSE TRACKER SUMMARY ===');
  console.log(`Balance: ₹${summary.balance}`);
  console.log(`Income: ₹${summary.income}`);
  console.log(`Expenses: ₹${summary.expense}`);
  console.log('\n=== ALL TRANSACTIONS ===');
  tracker.transactions.forEach(t =>
    console.log(`${t.date} | ${t.description} | ${t.type} | ₹${t.amount} (${t.category})`)
  );

  module.exports = ExpenseTracker;
}
