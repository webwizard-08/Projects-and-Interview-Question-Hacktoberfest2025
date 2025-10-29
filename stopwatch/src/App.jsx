import { useState,useEffect } from 'react'

import './App.css'

function App() {
  const [isCounting, setIsCounting] = useState(false)
  const [count, setCount] = useState(0)
  useEffect(()=>{
    if(!isCounting) return

    const interval = setInterval(() => {
      setCount((prev) => prev + 1)
    }, 1000);
    return () => {
      clearInterval(interval)
    }
  },[isCounting])

  return (
    <>
      <h1 className='text-3xl font-bold underline'>Stopwatch</h1>
      <div className='text-6xl'>{count}</div>
      <button
      onClick={()=>{
        setIsCounting((prev) => !prev)
      }}>
        {isCounting ? 'Stop' : 'Start'}
      </button>
    
    </>
  )
}

export default App
