import uuid

class MCPMessage:
    def __init__(self, sender, receiver, msg_type, payload):
        self.message = {
            "sender": sender,
            "receiver": receiver,
            "type": msg_type,
            "trace_id": str(uuid.uuid4()),
            "payload": payload,
        }

    def get(self):
        return self.message
