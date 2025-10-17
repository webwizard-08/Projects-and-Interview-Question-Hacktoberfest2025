// Basic React Component Examples

// 1. Simple Functional Component
function Welcome({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// 2. Component with State
function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        Increment
      </button>
    </div>
  );
}

// 3. Component with useEffect
function DataFetcher() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch('/api/data')
      .then(response => response.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error:', error);
        setLoading(false);
      });
  }, []);
  
  if (loading) return <div>Loading...</div>;
  return <div>{data ? data.message : 'No data'}</div>;
}

// 4. Custom Hook
function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);
  
  const increment = useCallback(() => setCount(c => c + 1), []);
  const decrement = useCallback(() => setCount(c => c - 1), []);
  const reset = useCallback(() => setCount(initialValue), [initialValue]);
  
  return { count, increment, decrement, reset };
}

// 5. Context Provider
const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  }, []);
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// 6. Context Consumer
function ThemeToggle() {
  const { theme, toggleTheme } = useContext(ThemeContext);
  
  return (
    <button onClick={toggleTheme}>
      Current theme: {theme}
    </button>
  );
}

// 7. Form with Validation
function ContactForm() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });
  const [errors, setErrors] = useState({});
  
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    
    if (!formData.message.trim()) {
      newErrors.message = 'Message is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      console.log('Form submitted:', formData);
    }
  };
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <div>
        <input
          type="text"
          name="name"
          value={formData.name}
          onChange={handleChange}
          placeholder="Name"
        />
        {errors.name && <span className="error">{errors.name}</span>}
      </div>
      
      <div>
        <input
          type="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          placeholder="Email"
        />
        {errors.email && <span className="error">{errors.email}</span>}
      </div>
      
      <div>
        <textarea
          name="message"
          value={formData.message}
          onChange={handleChange}
          placeholder="Message"
        />
        {errors.message && <span className="error">{errors.message}</span>}
      </div>
      
      <button type="submit">Submit</button>
    </form>
  );
}

// 8. List with Keys
function TodoList({ todos }) {
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>
          {todo.text}
        </li>
      ))}
    </ul>
  );
}

// 9. Conditional Rendering
function UserProfile({ user }) {
  if (!user) {
    return <div>Please log in</div>;
  }
  
  return (
    <div>
      <h2>Welcome, {user.name}!</h2>
      {user.isAdmin && <p>Admin privileges</p>}
    </div>
  );
}

// 10. Error Boundary
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }
    
    return this.props.children;
  }
}

export {
  Welcome,
  Counter,
  DataFetcher,
  useCounter,
  ThemeProvider,
  ThemeToggle,
  ContactForm,
  TodoList,
  UserProfile,
  ErrorBoundary
};
