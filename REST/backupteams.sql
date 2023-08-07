

--
-- Table structure for table `Teams`
--

DROP TABLE IF EXISTS `Teams`;

CREATE TABLE `Teams` (
  `team_id` int(11) NOT NULL AUTO_INCREMENT,
  `team_name` varchar(255) NOT NULL,
  `sport` varchar(255) NOT NULL,
  `average_age` int(11) DEFAULT NULL,
  PRIMARY KEY (`team_id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=latin1;


--
-- Dumping data for table `Teams`
--

LOCK TABLES `Teams` WRITE;

INSERT INTO `Teams` VALUES (1,'Airbenders','Frisbee',25),(2,'Duck Hunt','Frisbee',28),(3,'Frisboos','Frisbee',23);

UNLOCK TABLES;
