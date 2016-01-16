

        """
        for author in authors.split(';'):
            a = author.split(' ')
            s = ""
            for b in a:
                s += b
            #print s
            key = s
            if author_dict.has_key(key) is True:
                author_dict[key][int(topic_id)] += 1
            else:
                author_dict[key] = [0]*3
                author_dict[key][int(topic_id)] = 1

         

            """ 
            a = tuple(author.split(' '))
            print a 
            print author, topic_id
            """

           
        #find number of time each author occurs

#print author_dict

"""
