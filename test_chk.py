import tkinter as tk

def test():
    root = tk.Tk()
    v = tk.BooleanVar(value=True)
    
    # Check if this can be unchecked
    cb = tk.Checkbutton(root, text="Test", variable=v, onvalue=True, offvalue=False)
    cb.pack()

    def print_val():
        print("Value:", v.get())
        root.after(500, print_val)

    # Let's programmatically try to toggle it:
    cb.toggle()
    print("After toggle:", v.get())
    
    cb.toggle()
    print("After second toggle:", v.get())
    
    root.destroy()

test()
