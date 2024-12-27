function myMenuFunction() {
    var i = document.getElementById("navMenu");
    if(i.className === "nav-menu") {
        i.className += " responsive";
    } else {
        i.className = "nav-menu";
    }
}

function showLogin() {
    console.log("Switching to Login form");
    document.getElementById("login").style.left = "4px";
    document.getElementById("register").style.right = "-520px";
    document.getElementById("loginBtn").classList.add("white-btn");
    document.getElementById("registerBtn").classList.remove("white-btn");
}

function showRegister() {
    console.log("Switching to Register form");
    document.getElementById("login").style.left = "-510px";
    document.getElementById("register").style.right = "5px";
    document.getElementById("loginBtn").classList.remove("white-btn");
    document.getElementById("registerBtn").classList.add("white-btn");
}
