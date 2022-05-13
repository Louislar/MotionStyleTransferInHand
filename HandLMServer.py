from http.server import BaseHTTPRequestHandler, HTTPServer
from httpServer import MyServer, hostName, serverPort
import threading
import time

stopThread = False

class HandLMServer():
    def __init__(self, hostIP=None, hostPort=None) -> None:
        self.hostName=hostIP
        self.serverPort=hostPort
        self.webServer=None
        self.httpServerThread=None
        self.curSentMsg=['']    # List for pass by reference(Not sure if it works? -> it works :-))

    '''
    Reason of this is because we want to send a variable in the _requestHandler class, 
    Python did not allow inner class use instances in outer class!!
    '''
    def _requestHandlerClassFunc(self):
        msgStringNeedToSend = self.curSentMsg
        class _requestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(bytes(msgStringNeedToSend[0], "utf-8"))
        return _requestHandler

    def httpServerServeForever(self): 
        self.webServer.serve_forever()

    def startHTTPServerThread(self):
        self._requestHandler = self._requestHandlerClassFunc()
        self.webServer=HTTPServer((self.hostName, self.serverPort), self._requestHandler)
        print("Server started http://%s:%s" % (hostName, serverPort))
        httpServerThread = threading.Thread(target=self.httpServerServeForever,)
        httpServerThread.setDaemon(True)    # When Main thead ends, the sub thread will terminate
        httpServerThread.start()

    def stopHTTPServer(self):
        self.webServer.shutdown()
        self.webServer.server_close()
        print("Server stopped.")

if __name__=='__main__': 
    handLMServer = HandLMServer(hostIP=hostName, hostPort=serverPort)
    handLMServer.startHTTPServerThread()
    curTime = time.time()
    while True:
        try:
            if time.time() - curTime > 3: 
                handLMServer.curSentMsg[0] = str(curTime)
                curTime=time.time()
            pass
        except KeyboardInterrupt:
            print('keyboard interrupt')
            handLMServer.stopHTTPServer()


# def httpServerServeForever(server: HTTPServer):
#     try: 
#         server.serve_forever()
#     except KeyboardInterrupt: 
#         pass

# if __name__=='__main__': 
#     webServer = HTTPServer((hostName, serverPort), MyServer)
#     print("Server started http://%s:%s" % (hostName, serverPort))
#     httpServerThread = threading.Thread(target=httpServerServeForever, args=(webServer, ))
#     httpServerThread.setDaemon(True)    # When Main thead ends, the sub thread will terminate
#     httpServerThread.start()
#     # httpServerThread.join()    # Wait for the thread to terminate
#     while True:
#         try: 
#             pass
#         except KeyboardInterrupt: 
#             print('keyboard interrupt')
#             webServer.shutdown()
#             webServer.server_close()
#             print("Server stopped.")
#             break
#     # while not stopThread: 
#     #     pass