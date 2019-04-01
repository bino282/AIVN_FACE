from skpy import Skype
sk = Skype("nha28021995", "Bino@1995") # connect to Skype

ch = sk.contacts.contact["skype_username_where_we_want_send_message"].chat 

ch.sendMsg("some message") 